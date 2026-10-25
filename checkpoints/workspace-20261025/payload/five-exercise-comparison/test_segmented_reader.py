"""분할 TAR 경계 읽기를 실제 작은 TAR로 검사한다."""

import io
import tarfile
import unittest
from segmented_reader import SegmentedReader


class SegmentedReaderTests(unittest.TestCase):
    """분할 경계·누락·손상을 작은 합성 원본으로 검사한다."""

    def test_cross_segment_seek(self):
        """사이 포장 바이트를 제외하고 경계를 넘어 읽고 끝으로 이동한다."""
        source = io.BytesIO(b"PADabcPADdefgPADhi")
        with SegmentedReader(
            source, [(0, 3, 3), (3, 9, 4), (7, 16, 2)]
        ) as stream:
            self.assertEqual(stream.read(5), b"abcde")
            stream.seek(-3, io.SEEK_END)
            self.assertEqual(stream.read(), b"ghi")
            self.assertEqual(stream.read(1), b"")

    def test_missing_segment_rejected(self):
        """논리 주소 누락과 물리 구간 절단을 거부한다."""
        with self.assertRaises(ValueError):
            SegmentedReader(io.BytesIO(b"abcd"), [(0, 0, 2), (3, 2, 2)])
        with self.assertRaises(OSError):
            SegmentedReader(io.BytesIO(b"ab"), [(0, 0, 3)]).read()

    def test_nested_tar_across_headers(self):
        """TAR 헤더와 본문 중간이 분할돼도 원래 멤버를 읽는다."""
        original = io.BytesIO()
        with tarfile.open(fileobj=original, mode="w") as archive:
            item = tarfile.TarInfo("Day01/a.jpg")
            item.size = 7
            archive.addfile(item, io.BytesIO(b"example"))
        data = original.getvalue()
        cut = 515
        wrapped = io.BytesIO(b"WRAP" + data[:cut] + b"HEADER" + data[cut:])
        with SegmentedReader(
            wrapped, [(0, 4, cut), (cut, 4 + cut + 6, len(data) - cut)]
        ) as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                self.assertEqual(
                    archive.extractfile("Day01/a.jpg").read(), b"example"
                )


if __name__ == "__main__":
    unittest.main()
