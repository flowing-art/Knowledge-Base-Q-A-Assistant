"""上传和删除要口令；过大文件 413；错误信息里不出现本机路径。"""

import os
import unittest

from fastapi.testclient import TestClient

import app as app_module

# 假装成服务器上的真实路径：这类内容不允许出现在返回给浏览器的信息里。
FAKE_LOCAL_PATH = r"C:\private\kb\secret.txt"
LEAK_MARKERS = ("private", "secret.txt")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self._token = os.environ.get("KB_TOKEN")
        os.environ["KB_TOKEN"] = "rumen-local"
        self.client = TestClient(app_module.app)

    def tearDown(self):
        if self._token is None:
            os.environ.pop("KB_TOKEN", None)
        else:
            os.environ["KB_TOKEN"] = self._token

    def assertNoLocalPath(self, text):
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, text)

    def test_public_error_hides_local_path(self):
        text = app_module.public_error("回答", RuntimeError(FAKE_LOCAL_PATH))
        self.assertNoLocalPath(text)
        self.assertIn("终端日志", text)

    def test_upload_and_delete_reject_bad_token(self):
        upload = self.client.post(
            "/upload",
            files={"file": ("a.txt", b"hello", "text/plain")},
        )
        self.assertEqual(upload.status_code, 401)

        wrong = self.client.post(
            "/documents/delete",
            json={"name": "a.txt"},
            headers={"X-KB-Token": "nope"},
        )
        self.assertEqual(wrong.status_code, 401)

        session = self.client.delete("/sessions/does-not-exist")
        self.assertEqual(session.status_code, 401)

    def test_missing_token_config_is_503(self):
        os.environ["KB_TOKEN"] = ""
        response = self.client.post("/documents/delete", json={"name": "a.txt"})
        self.assertEqual(response.status_code, 503)
        self.assertNoLocalPath(response.text)

    def test_oversize_upload_is_413(self):
        original = app_module.max_upload_bytes
        app_module.max_upload_bytes = lambda: 4
        try:
            response = self.client.post(
                "/upload",
                files={"file": ("a.txt", b"hello", "text/plain")},
                headers={"X-KB-Token": "rumen-local"},
            )
        finally:
            app_module.max_upload_bytes = original
        self.assertEqual(response.status_code, 413)
        self.assertNoLocalPath(response.text)


if __name__ == "__main__":
    unittest.main()
