# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl.utils.device import logical_index_from_visible_devices


def test_non_zero_visible_devices_map_to_logical_index():
    assert logical_index_from_visible_devices("2", visible_devices="2,3") == "0"
    assert logical_index_from_visible_devices("3", visible_devices="2,3") == "1"
    assert logical_index_from_visible_devices(2, visible_devices="2, 3") == "0"


def test_zero_based_visible_devices_keep_the_same_index():
    assert logical_index_from_visible_devices("2", visible_devices="0,1,2,3") == "2"
    assert logical_index_from_visible_devices("0", visible_devices="0,1") == "0"


def test_missing_visible_devices_keep_the_accelerator_id():
    assert logical_index_from_visible_devices("2", visible_devices="") == "2"
    assert logical_index_from_visible_devices("4", visible_devices="2,3") == "4"


def test_visible_devices_are_read_from_the_environment(monkeypatch):
    monkeypatch.setattr("verl.utils.device.get_visible_devices_keyword", lambda: "ASCEND_RT_VISIBLE_DEVICES")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    assert logical_index_from_visible_devices("3") == "1"
