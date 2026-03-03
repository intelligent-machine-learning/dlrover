# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from dlrover.trainer.util.conf_util import (
    ConfigurationManagerInterface,
    ConfigurationManagerMeta,
    get_conf,
)


class NewConfigurationManagerTest(unittest.TestCase):
    def tearDown(self):
        ConfigurationManagerMeta._all_conf_by_name = {}
        ConfigurationManagerMeta._final_attrs = {}
        ConfigurationManagerMeta._override_order = []
        ConfigurationManagerMeta._default_priority = 1 << 30

    def test_priority(self):
        class a(ConfigurationManagerInterface):
            batch_size = 1
            priority = 10
            thisa = "aaa"
            train_set = {"path": "23", "sls": {"ak": "1", "sk": "2"}}

        class b(ConfigurationManagerInterface):
            batch_size = 10
            priority = 9
            thisb = "bbb"
            train_set = {"path": "231", "sls": {"ak": "1", "sk": "44"}}

        final_conf = ConfigurationManagerMeta.merge_configs()
        self.assertDictEqual(final_conf.train_set, b.train_set)
        self.assertEqual(10, final_conf.batch_size)
        self.assertEqual("bbb", final_conf.thisb)
        self.assertEqual("aaa", final_conf.get("thisa"))
        # test base and meta method
        self.assertIsNone(final_conf.get("this_not_exist"))
        self.assertIsNone(final_conf["this_not_exist"])
        self.assertFalse("this_not_exist" in final_conf)
        final_conf["this_not_exist"] = "haha"
        self.assertEqual("haha", final_conf["this_not_exist"])
        self.assertTrue("this_not_exist" in final_conf)

    def test_build(self):
        class rules(ConfigurationManagerInterface):
            priority = 1
            __path = """
            {
                "rules": [
                    {
                        "metric": "auc",
                        "op": ">",
                        "val": "0.0001"
                    }
                ],
                "steps": 40,
                "export": {
                    "model_dir": [
                        "pangu://pangu1_analyze_sata_em14_online/"
                    ],
                    "meta_file_for_arks": "test.txt",
                    "protocol": "arks"
                }
            }
            """

            @classmethod
            def build(cls):
                import json

                rules = json.loads(cls.__path)
                cls.rules = rules
                cls.test_set = {"rules": rules}

        class test_set(ConfigurationManagerInterface):
            a = 1
            priority = 11
            test_set = {"path": "231", "sls": {"ak": "1", "sk": "44"}}

        final_conf = ConfigurationManagerMeta.merge_configs()
        self.assertDictEqual(final_conf.test_set["rules"], rules.rules)
        self.assertEqual(
            "test_set<rules",
            "<".join(ConfigurationManagerMeta._override_order),
        )

    def test_mutable(self):
        class a(ConfigurationManagerInterface):
            batch_size = 1
            priority = 10
            thisa = "aaa"
            train_set = {"path": "23", "sls": {"ak": "1", "sk": "2"}}

        final_conf = ConfigurationManagerMeta.merge_configs()
        train_set = final_conf.get("train_set")
        self.assertDictEqual(train_set, a.train_set)

        train_set["path"] = "100"
        new_train_set = {"path": "100", "sls": {"ak": "1", "sk": "2"}}
        self.assertDictEqual(final_conf.get("train_set"), new_train_set)

        test_set = final_conf.get("test_set", {})
        self.assertDictEqual(test_set, {})
        self.assertNotIn("test_set", final_conf)

    def test_register(self):
        class my1:
            b = "my1"

        class my2:
            b = "my2"

        ConfigurationManagerMeta.register(my1)
        ConfigurationManagerMeta.register(my2)
        final_conf = ConfigurationManagerMeta.merge_configs()
        self.assertEqual(final_conf.b, "my2")

    def test_default_priority(self):
        class test_set_1(ConfigurationManagerInterface):
            a = 1
            test_set = {"path": "231", "sls": {"ak": "1", "sk": "44"}}

        class test_set_2(ConfigurationManagerInterface):
            a = 22
            test_set = {"path": "dd", "sls": {"ak": "23", "sk": "ss"}}

        final_conf = ConfigurationManagerMeta.merge_configs()
        self.assertEqual(final_conf.a, 22)
        self.assertDictEqual(final_conf.test_set, test_set_2.test_set)

    def test_clear(self):
        class test_set_1(ConfigurationManagerInterface):
            a = 1
            test_set = {"path": "231", "sls": {"ak": "1", "sk": "44"}}

        class test_set_2(ConfigurationManagerInterface):
            a = 22
            test_set = {"path": "dd", "sls": {"ak": "23", "sk": "ss"}}

        final_conf = ConfigurationManagerMeta.merge_configs()
        self.assertTrue(final_conf.get("a") == 22)
        final_conf.clear()
        self.assertEqual(str(final_conf), "{}")

    def test_get_conf(self):
        class TrainConf(object):
            batch_size = 64
            log_steps = 100
            save_steps = 1000
            save_min_secs = 60
            save_max_secs = 60 * 6

            params = {
                "deep_embedding_dim": 8,
                "learning_rate": 0.0001,
                "l1": 0.0,
                "l21": 0.0,
                "l2": 0.0,
                "optimizer": "group_adam",
                "log_steps": 100,
            }

        conf = get_conf(TrainConf)
        self.assertTrue(conf.get("batch_size") == 64)
        self.assertTrue(conf.get("epoch") is None)
        self.assertTrue(conf.get("epoch", 1) == 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
