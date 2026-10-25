"""원시 TAR 경로 대조 시 안전한 상대 이미지 경로만 사용하는지 검사한다."""

import unittest
from inspect_aihub_raw import canonical_image_path


class RawPathTests(unittest.TestCase):
    """라벨 키 대조에서 루트 접두어와 경로 이동을 구분한다."""

    def test_root_prefix_removed(self):
        """서버 포장 폴더가 달라도 Day 이후의 같은 이미지를 찾는다."""
        self.assertEqual(
            canonical_image_path("valid/Day32_201104_F/1/A/example.jpg"),
            "Day32_201104_F/1/A/example.jpg",
        )
        self.assertIsNone(canonical_image_path("valid/info.json"))

    def test_unsafe_path_rejected(self):
        """원시 압축에 외부 경로가 있으면 추출 후보로 허용하지 않는다."""
        for path in ("/Day01/a.jpg", "../Day01/a.jpg", "Day01/../../a.jpg"):
            with self.assertRaises(ValueError):
                canonical_image_path(path)


if __name__ == "__main__":
    unittest.main()
