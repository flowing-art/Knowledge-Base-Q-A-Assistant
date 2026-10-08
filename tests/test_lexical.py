"""重排分数不够时，字面二元组兜底该捞回的捞回、无关的仍丢掉。"""

import unittest

import rag


class _Doc:
    def __init__(self, text):
        self.page_content = text


SOFT_DOC = "软件名称 低光图像综合增强系统 第一开发者：王耀政 其他开发者：无"


class LexicalRescueTests(unittest.TestCase):
    def test_bigrams_drop_punctuation(self):
        self.assertEqual(
            rag._bigrams("开发者都有谁"),
            ["开发", "发者", "者都", "都有", "有谁"],
        )

    def test_low_rerank_but_same_words_is_rescued(self):
        found = rag._lexical_rescue("开发者都有谁", [(_Doc(SOFT_DOC), 0.026)])
        self.assertEqual(len(found), 1)
        self.assertGreaterEqual(found[0][2], rag.LEXICAL_MIN_HITS)

    def test_unrelated_question_stays_empty(self):
        found = rag._lexical_rescue("今天天气怎么样", [(_Doc(SOFT_DOC), 0.0)])
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
