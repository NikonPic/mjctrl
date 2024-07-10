import mujoco
import mujoco.viewer
import numpy as np
import time

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1
Kpos: float = 0.7
Kori: float = 0.7

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the impedance control.
K_x_values = np.array([300.0, 300.0, 300.0, 10.0, 10.0, 10.0])
K_x = np.diag(K_x_values)  # Stiffness matrix
D_x = np.diag(2.0 * np.sqrt(K_x_values))  # Damping matrix

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint torque.
max_tau = 20


def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y])


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id

    # Get the dof and actuator ids for the joints we wish to control.
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
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    x_dot = np.zeros(6)
    eye = np.eye(model.nv)

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

            # Set the desired target position and orientation.
            data.mocap_pos[mocap_id, 0:2] = circle(
                2*data.time, 0.1, 0.5, 0.0, 0.5)
            x_d = data.mocap_pos[mocap_id]
            quat_d = data.mocap_quat[mocap_id]

            # Current end-effector position and orientation.
            x = data.site(site_id).xpos
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, quat_d, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)

            # Position and orientation errors.
            dx = x_d - x
            twist[:3] = Kpos * dx / integration_dt
            twist[3:] *= Kori / integration_dt

            # Current end-effector velocity.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jac_reduced = jac[:, dof_ids]
            x_dot = jac @ data.qvel

            # Desired velocity (zero in this case).
            x_dot_d = np.zeros(6)

            # Impedance control law.
            F = K_x @ twist + D_x @ (x_dot_d - x_dot)

            # Solve for joint torques.
            tau_imp = jac_reduced.T @ F
            tau_np = (eye - np.linalg.pinv(jac) @
                      jac)[dof_ids, dof_ids] @ (Kn * (q0 - data.qpos[dof_ids]))
            tau = tau_imp + tau_np

            tau_max_cur = np.abs(tau).max()
            if tau_max_cur > max_tau:
                tau *= max_tau / tau_max_cur

            data.ctrl[actuator_ids] = tau[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
