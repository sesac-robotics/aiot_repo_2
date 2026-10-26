"""온도 보호 재개 조건과 연속 냉각 대기를 실제 서비스 조작 없이 검증한다."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import safety_guard as guard


class AutoResumeChecks(unittest.TestCase):
    """센서 누락·전원 분리·냉각 도중 재가열에 대한 자동 재개 차단을 검사한다."""

    def test_resume_requires_both_sensors_and_power(self):
        """CPU·GPU 상한과 전원을 모두 만족할 때만 재개 후보로 인정한다."""
        good=dict(cpu_c=80,gpu_c=65,ac_connected=True)
        self.assertTrue(guard.resume_ready(good))
        for delta in [dict(cpu_c=81),dict(gpu_c=66),dict(cpu_c=None),dict(gpu_c=None),dict(ac_connected=False)]:
            self.assertFalse(guard.resume_ready(dict(good,**delta)))

    def test_reheating_resets_continuous_cooling(self):
        """120초 안에 재가열되면 대기를 처음부터 계산하고 성공 후 표식을 지운다."""
        good=dict(cpu_c=70,gpu_c=50,ac_connected=True)
        readings=[good,good,dict(good,cpu_c=90),good,good,good]
        with tempfile.TemporaryDirectory() as folder:
            pending=Path(folder)/'pending.json';pending.write_text('{}')
            with patch.object(guard,'OUT',Path(folder)),patch.object(guard,'PENDING',pending),patch.object(guard,'snapshot',side_effect=readings),patch.object(guard.time,'monotonic',side_effect=[0,60,120,180,240,300]),patch.object(guard.time,'sleep'),patch.object(guard.subprocess,'run') as run:
                guard.await_cooling()
                run.assert_called_once_with(['systemctl','--user','start',guard.UNIT],check=True,timeout=45)
                self.assertFalse(pending.exists())
                self.assertTrue((Path(folder)/'safety_resume.json').exists())


if __name__=='__main__':unittest.main()
