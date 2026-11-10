from scripts.audit_resources import container_limits


def test_container_quota_overrides_host_capacity(tmp_path):
    (tmp_path / "memory.max").write_text(str(80 * 2**30))
    (tmp_path / "memory.current").write_text(str(7 * 2**30))
    (tmp_path / "cpu.max").write_text("1600000 100000")
    limits = container_limits(tmp_path)
    assert limits["memory.max"] == 80 * 2**30
    assert limits["cpu_quota_cores"] == 16


def test_unlimited_cgroup_is_not_a_numeric_budget(tmp_path):
    (tmp_path / "memory.max").write_text("max")
    (tmp_path / "cpu.max").write_text("max 100000")
    limits = container_limits(tmp_path)
    assert limits["memory.max"] is None
    assert limits["cpu_quota_cores"] is None
