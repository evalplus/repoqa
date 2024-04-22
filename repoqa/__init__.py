# SPDX-FileCopyrightText: (c) 2024 EvalPlus Team
#
# SPDX-License-Identifier: Apache-2.0

try:
    from repoqa._version import __version__, __version_tuple__
except ImportError:
    __version__ = "local-dev"
