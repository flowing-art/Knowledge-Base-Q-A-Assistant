"""会话历史：建会话、记问答、删除。用临时库，不碰项目里的 history.db。"""

import os
import tempfile
import unittest

import history


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self._old_path = history.DB_PATH
        self._old_inited = history._inited
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.path = path
        history.DB_PATH = path
        history._inited = False

    def tearDown(self):
        history.DB_PATH = self._old_path
        history._inited = self._old_inited
        try:
            os.remove(self.path)
        except OSError:
            pass

    def test_record_list_and_delete(self):
        history.record_turn("s1", "开发者是谁", "王耀政")
        history.record_turn("s1", "还有呢", "没有其他开发者")

        sessions = history.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], "s1")
        self.assertEqual(sessions[0]["message_count"], 4)
        self.assertIn("开发者", sessions[0]["title"])

        messages = history.get_messages("s1")
        self.assertEqual(
            [item["role"] for item in messages],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertEqual(history.delete_session("s1"), 4)
        self.assertEqual(history.list_sessions(), [])
        self.assertIsNone(history.get_session("s1"))


if __name__ == "__main__":
    unittest.main()
