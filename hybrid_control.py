import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# path to the model
model_path = "franka_knee/scene.xml"

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

# maximum time
end_time = 10

# desired force
F_des = [0, 0, -10]

# weight for movement control
twist_weight = 0.3


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])


def get_sensor_data(data, model, sensor_name):
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr:sensor_adr + sensor_dim]


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    site_name_ft = "attachment_site2"
    site_id_ft = model.site(site_name_ft).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos[dof_ids]


    # Mocap body we will control with our mouse.
    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Lists to collect force and torque data
    force_data_list = []
    torque_data_list = []
    time_list = []

    # Weighting factors for combining controllers
    load_minimization_weight = 1 - twist_weight

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            step_start = time.time()
            if data.time > end_time:
                break

            # Compute inverse dynamics to get required torques for given state
            mujoco.mj_inverse(model, data)

            # Get the joint torques
            joint_torques = data.qfrc_inverse[dof_ids]

            # Get the Jacobian at the end-effector
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Compute the end-effector force and torque using the Jacobian
            end_effector_load = jac[:, dof_ids] @ joint_torques[dof_ids]
            end_effector_forces = get_sensor_data(data, model, "force_eff")
            end_effector_torques = get_sensor_data(data, model, "torque")

            # Spatial velocity (aka twist).
            dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(
                error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Damped least squares for twist control
            dq_twist = (jac.T @ np.linalg.solve(jac @
                        jac.T + diag, twist))[dof_ids]
            # Nullspace control biasing joint velocities towards the home configuration.
            dq_twist_null = (eye - np.linalg.pinv(jac) @
                             jac)[dof_ids, dof_ids] @ (Kn * (q0 - data.qpos[dof_ids]))
            
            null_space_projector = eye - np.linalg.pinv(jac) @ jac
            
            # now calc jac with respect to ft sensor
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id_ft)


            F_actual = get_sensor_data(data, model, "force_eff")
            # dF
            F_control = 10 * (F_des - F_actual)
            torque_control_force = jac[:3, :].T @ F_control
            torque_control_null_space = null_space_projector[dof_ids, dof_ids] @ torque_control_force[dof_ids]

            # Combine controllers
            dq_combined = twist_weight * \
                (dq_twist + dq_twist_null) + load_minimization_weight * torque_control_null_space

            # Clamp maximum joint velocity.
            dq_abs_max = np.abs(dq_combined).max()
            if dq_abs_max > max_angvel:
                dq_combined *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()  # Note the copy here is important.
            dq_full = data.qvel.copy()
            dq_full[dof_ids] = dq_combined
            mujoco.mj_integratePos(model, q, dq_full, integration_dt)
            np.clip(q[:10], *model.jnt_range.T, out=q[:10])

            # Set the control signal and step the simulation.
            data.ctrl[actuator_ids] = q[dof_ids]
            mujoco.mj_step(model, data)

            force_data_list.append(end_effector_forces.copy())
            torque_data_list.append(end_effector_torques.copy())
            time_list.append(data.time)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # Convert lists to numpy arrays
        force_data_array = np.array(force_data_list)
        torque_data_array = np.array(torque_data_list)
        time_array = np.array(time_list)

        # Plot force and torque data
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_array, force_data_array)
        plt.title("Force at Attachment Site")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend(['Fx', 'Fy', 'Fz'])

        plt.subplot(2, 1, 2)
        plt.plot(time_array, torque_data_array)
        plt.title("Torque at Attachment Site")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.legend(['Tx', 'Ty', 'Tz'])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
