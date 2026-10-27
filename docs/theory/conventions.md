# Coordinates and units

Most sign and unit errors in motion code come from mixing frames. Every formula
in this repo states which frame it is in, using these symbols only.

| Symbol | Meaning | Unit |
| --- | --- | --- |
| `theta_M` | cam master-axis angle | rad |
| `omega_M` | cam master-axis speed (assumed constant, see below) | rad/s |
| `s` | slave linear displacement | m |
| `q` | generalized coordinate: motor angle (1 axis) or joint vector (robot) | rad |
| `k` | linear-to-motor conversion: `lead / 2*pi * gear_ratio` | m/rad |
| `tau` | torque at the motor / joint | N·m |
| `J_eq` | load inertia referred to the motor shaft | kg·m^2 |

## Cam chain

    q = s(theta_M) / k
    ds  = s'(theta_M)  * omega_M
    dds = s''(theta_M) * omega_M^2

**`omega_M` constant is a first-stage assumption.** Real synchronized operation
follows an encoder and the master can accelerate; then `dds` gains a
`s' * domega_M` term. State the assumption wherever results depend on it.

## Robots

Robots are handled directly in `q`: there is no `k`, and the dynamics return
joint torque. The difference between a cam axis and a robot is absorbed by the
`LoadModel` implementation, not by the optimizer.

## Rule

A `LoadModel` takes `q, dq, ddq` and nothing else. If an implementation needs the
master angle, it converts internally. Callers never do the conversion.
