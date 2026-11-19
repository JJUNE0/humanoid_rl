"""Use the same fixed mjlab scene and expose the reference clock to the actor."""

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from humanoid_rl.env.imitation.amp import configure_amp_environment


def reference_phase(env, num_frames):
    return (env.episode_length_buf.float() / (num_frames-1)).clamp(0, 1).unsqueeze(-1)


def reference_finished(env, num_frames):
    return env.episode_length_buf >= num_frames-1


def configure_add_environment(cfg, add_config, *, fixed_scene=None, controller=None):
    configure_amp_environment(cfg, amp_config=add_config, fixed_scene=fixed_scene, controller=controller)
    frames = cfg.fixed_scene["frames"]
    cfg.add_tracking = True
    cfg.observations["actor"].terms["reference_phase"] = ObservationTermCfg(
        func=reference_phase, params={"num_frames": frames})
    # Reference completion is a true terminal state: do not bootstrap beyond the clip.
    cfg.terminations["reference_finished"] = TerminationTermCfg(
        func=reference_finished, params={"num_frames": frames})
    cfg.episode_length_s = min(cfg.episode_length_s, (frames-1)*cfg.sim.mujoco.timestep*cfg.decimation)
