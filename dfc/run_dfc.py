#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany, Maria Cervera
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :run_dfc.py
# @author         :am
# @contact        :ameulema@ethz.ch
# @created        :25/11/2021
# @version        :1.0
# @python_version :3.6.8
"""
Main script for training networks with deep feedback control
------------------------------------------------------------

This script is used for training networks on a certain dataset according to
the deep feedback control algorithm.
"""
from main import run

if __name__=='__main__':
    run(network_type='DFC')
