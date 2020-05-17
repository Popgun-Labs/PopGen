#!/usr/bin/env bash
autopep8 -i -r --global-config ./tox.ini *.py
autopep8 -i -r --global-config ./tox.ini popgen/*