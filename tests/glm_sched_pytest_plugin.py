"""pytest plugin: run vLLM's own scheduler tests with the GLM_PREFILL_SCHED edits installed.

The knobs stay off (GLM_PREFILL_* unset), so every vendor test must still pass: the edited
Scheduler.schedule has to reduce to the stock one. From a vLLM 487ecf187 tree:

  PYTHONPATH=.:<glm repo>/overlay:<glm repo>/tests python -m pytest --noconftest \
      -p glm_sched_pytest_plugin tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py
"""


def pytest_configure(config):
    import glm_prefill_sched as gps
    import vllm.v1.core.sched.scheduler as smod

    assert not (gps.CADENCE or gps.SHORT_TOKENS or gps.WHEN_QUEUED or gps.END_DRAIN), "knobs must be off"
    smod._glm_ps = gps
    smod.Scheduler.schedule = gps.build_schedule(smod)
    config._glm_patched = True


def pytest_report_header(config):
    return "GLM_PREFILL_SCHED edits installed on Scheduler.schedule (knobs off)"
