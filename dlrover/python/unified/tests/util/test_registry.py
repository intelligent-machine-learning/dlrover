#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from dlrover.python.unified.controller.schedule.scaler import BaseRayNodeScaler
from dlrover.python.unified.util.auto_registry import AutoExtensionRegistry


@pytest.fixture
def tmp_extension_project():
    extension_project = "project_b"
    p = Path(f"{extension_project}/__init__.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")

    p = Path(f"{extension_project}/extensions/__init__.py")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")

    with open(f"{extension_project}/extensions/test_scaler.py", "w") as f:
        f.write("""                                               
import sys
sys.path.insert(0, "..")                        
from dlrover.python.unified.controller.schedule.scaler import BaseRayNodeScaler
from dlrover.python.unified.util.auto_registry import extension          

@extension(priority=10)
class TestScaler(BaseRayNodeScaler):                       

    def scale_up(self, count: int):                            
        return [f"test-{i}" for i in range(count * 2)] 

    def scale_down(self, node_ids):                            
        print(f"test scaling down: {node_ids}")          
        return True                                            
            """)

    yield extension_project

    shutil.rmtree(f"{extension_project}")


def test_extension_registry(tmp_extension_project):
    interface_name = (
        f"{BaseRayNodeScaler.__module__}.{BaseRayNodeScaler.__qualname__}"
    )

    assert (
        AutoExtensionRegistry.get_original_class_by_interface(interface_name)
        is None
    )

    AutoExtensionRegistry.auto_discover(tmp_extension_project)
    assert len(AutoExtensionRegistry._original_impl) == 1
    assert len(AutoExtensionRegistry._extension_impl) == 1

    assert (
        AutoExtensionRegistry.get_original_class_by_interface(interface_name)
        is not None
    )


@patch.object(AutoExtensionRegistry, "_scan_package_for_extensions")
@patch.object(AutoExtensionRegistry, "_scanned_modules")
def test_invalid_extension_registry(
    mock_scanned_modules, mock_scan_package_for_extensions
):
    mock_scanned_modules.side_effect = ImportError()
    AutoExtensionRegistry.auto_discover("test123")

    mock_scan_package_for_extensions.side_effect = Exception()
    with pytest.raises(Exception):
        AutoExtensionRegistry.auto_discover("test123")
