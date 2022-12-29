import unittest 

from trainer.util.conf_util import ConfigurationManagerMeta, ConfigurationManagerInterface

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
        final_conf.clear()
        self.assertEqual(str(final_conf), "{}")


if __name__ == "__main__":
    unittest.main(verbosity=2)