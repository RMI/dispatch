Cheshire: a Python Template Repository for Catalyst
=======================================================================================

.. readme-intro

.. image:: https://github.com/catalyst-cooperative/cheshire/workflows/tox-pytest/badge.svg
   :target: https://github.com/catalyst-cooperative/cheshire/actions?query=workflow%3Atox-pytest
   :alt: Tox-PyTest Status

.. image:: https://github.com/catalyst-cooperative/cheshire/workflows/repo2docker/badge.svg
   :target: https://github.com/catalyst-cooperative/cheshire/actions?query=workflow%3Arepo2docker
   :alt: repo2docker Build Status

.. image:: https://github.com/catalyst-cooperative/cheshire/workflows/docker-build-push/badge.svg
   :target: https://github.com/catalyst-cooperative/cheshire/actions?query=workflow%3Adocker-build-push
   :alt: Docker build status

.. image:: https://img.shields.io/codecov/c/github/catalyst-cooperative/cheshire?style=flat&logo=codecov
   :target: https://codecov.io/gh/catalyst-cooperative/cheshire
   :alt: Codecov Test Coverage

.. image:: https://img.shields.io/readthedocs/catalystcoop-cheshire?style=flat&logo=readthedocs
   :target: https://catalystcoop-cheshire.readthedocs.io/en/latest/
   :alt: Read the Docs Build Status

.. image:: https://img.shields.io/pypi/v/catalystcoop.cheshire?style=flat&logo=python
   :target: https://pypi.org/project/catalystcoop.cheshire/
   :alt: PyPI Latest Version

.. image:: https://img.shields.io/conda/vn/conda-forge/catalystcoop.cheshire?style=flat&logo=condaforge
   :target: https://anaconda.org/conda-forge/catalystcoop.cheshire
   :alt: conda-forge Version

.. image:: https://img.shields.io/pypi/pyversions/catalystcoop.cheshire?style=flat&logo=python
   :target: https://pypi.org/project/catalystcoop.cheshire/
   :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black>
   :alt: Any color you want, so long as it's black.

This template repository helps make new Python projects easier to set up and more
uniform. It contains a lot of infrastructure surrounding a minimal Python package named
``cheshire`` (the cat who isn't entirely there...).

Create a new repository from this template
=======================================================================================

* Choose a name for the new package that you are creating.
* The name of the repository should be the same as the name of the new Python package
  you are going to create. E.g. a repository at ``catalyst-cooperative/cheshire`` should
  be used to define a package named ``cheshire``.
* Fork this template repository to create a new Python project repo.
  `See these instructions <https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template>`__.
* Clone the new repository to your development machine.
* Run ``pre-commit install`` in the newly clone repository to install the
  `pre-commit hooks <https://pre-commit.com/>`__ defined in ``.pre-commit-config.yaml``
* Create the ``cheshire`` conda environment by running ``conda env create`` or
  (preferably) ``mamba env create`` in the top level of the repository.
* Activate the new conda environment with ``conda activate cheshire``.
* Run ``tox`` from the top level of the repository to verify that everything is working
  correctly.

Rename the package and distribution
=======================================================================================

Once you know that your forked version of the ``cheshire`` package is working as
expected, you should update the package and distribution names in your new repo to
reflect the name of your new package. The **package name** is determined by the name of
the directory under ``src/`` which contains the source code, and is the name you'll use
to import the package for use in a program, script, or notebook. E.g.:

.. code:: python

  import cheshire

The **distribution name** is the name that is used to install the software using a
program like  ``pip``, ``conda``, or ``mamba``. It is often identical to the package
name, but can also contain a prefix namespace that indicates the individual or
organization responsible for maintaining the pacakge. See :pep:`423`
`PEP 423 <https://peps.python.org/pep-0423/>`__ for more on Python package naming
conventions. We are using the ``catalystcoop`` namespace for the packages that we
publish, so our ``pudl`` package becomes ``catalystcoop.pudl`` in the
Python Package Index (PyPI) or on ``conda-forge``. Similarly the ``cheshire`` package
becomes the ``catalystcoop.cheshire`` distribution. The distribution name is determined
by the ``name`` argument in the call to ``setup()`` in ``setup.py``.

.. code:: bash

  pip install catalystcoop.cheshire

The package and distribution names are referenced in many of the files in the template
repository, and they all need to be replaced with the name of your new package. You can
use ``grep -r`` to search recursively through all of the files for the word ``cheshire``
at the command line, or use the search-and-replace functionality of your IDE / text
editor. The name of the package directory under ``src/`` will also need to be changed.

* Supply any required tokens, e.g. for CodeCov
* Rename the ``src/cheshire`` directory to reflect the new package name.
* Search for ``cheshire`` and replace it as appropriate everywhere. Sometimes
  this will be with a distribution name like ``catalystcoop.cheshire``
  (the package as it appears for ``pip`` or ``PyPI``) and sometimes this will be the
  importable package name (the name of the directory under ``src`` e.g. ``cheshire``)
* Create the new project / package at Read The Docs.

What this template provides
=======================================================================================

Python Package Skeleton
-----------------------
* The ``src`` directory contains the code that will be packaged and deployed on the user
  system. That code is in a directory with the same name as the package.
* Using a separate ``src`` directory helps avoid accidentally importing the package when
  you're working in the top level directory of the repository.
* A simple python module (``dummy.py``), and a separate module providing a command line
  interface to that module (``cli.py``) are included as examples.
* Any files in the ``src/package_data/`` directory will also be packaged and deployed.
* What files are included in or excluded from the package on the user's system is
  controlled by the ``MANIFEST.in`` file and some options in the call to ``setup()`` in
  ``setup.py``.
* The CLI is deployed using a ``console_script`` entrypoint defined in ``setup.py``.
* We use ``setuptools_scm`` to obtain the package's version directly from ``git`` tags,
  rather than storing it in the repository and manually updating it.
* ``README.rst`` is read in and used for the pacakge's ``long_description``. This is
  what is displayed on the PyPI page for the package. For example, see the
  `PUDL Catalog <https://pypi.org/project/catalystcoop.pudl-catalog/0.1.0/>`__ page.
* By default we create at least three sets of "extras" -- additional optional package
  dependencies that can be installed in special circumstances: ``dev``, ``docs```, and
  ``tests``. The packages listed there are used in development, building the docs, and
  running the tests (respectively) but aren't required for a normal user who is just
  installing the package from ``pip`` or ``conda``.
* Python has recently evolved a more diverse community of build and packaging tools.
  Which flavor is being used by a given package is indicated by the contents of
  ``pyproject.toml``. That file also contains configuration for a few other tools,
  including ``black`` and ``isort``, described in the section on linters and formatters
  below.

Pytest Testing Framework
------------------------
* A skeleton `pytest <https://docs.pytest.org/>`_ testing setup is included in the
  ``tests/`` directory.
* Tests are split into ``unit`` and ``integration`` categories.
* Session-wide test fixtures, additional command line options, and other pytest
  configuration can be added to ``tests/conftest.py``
* Exactly what pytest commands are run during continuous integration controlled by Tox.
* Pytest can also be run manually without using Tox, but will use whatever your
  personal python environment happens to be, rather than the one specified by the
  package. Running pytest on its own is a good way to debug new or failing tests
  quickly, but we should always use Tox and its virtual environment for actual testing.

Test Coordination with Tox
--------------------------
* We define several different test environments for use with Tox in ``tox.ini``
* `Tox <https://tox.wiki/en/latest/>`__ is used to run pytest in an isolated Python
  virtual environment.
* We also use Tox to coordinate running the code linters, building the documentation,
  and releasing the software to PyPI.
* The default Tox environment is named ``ci`` and it will run the linters, build the
  documentation, run all the tests, and generate test coverage statistics.
* ``tox.ini`` also contains sections near the bottom which configure the behavior of
  ``doc8``, ``flake8``, ``pytest``, and ``rstcheck``.

Git Pre-commit Hooks
--------------------
* A variety of sanity checks are defined as git pre-commit hooks -- they run any time
  you try to make a commit, to catch common issues before they are saved. Many of these
  hooks are taken from the excellent `pre-commit project <https://pre-commit.com/>`__.
* The hooks are configured in ``.pre-commit-config.yaml``
* For them to run automatically when you try to make a commit, you **must** install the
  pre-commit hooks in your cloned repository first. This only has to be done once.
* These checks are run as part of our CI, and the CI will fail if the pre-commit hooks
  fail.
* We also use the `pre-commit.ci <https://pre-commit.ci>`__ service to run the same
  checks on any code that is pushed to GitHub, and to apply standard code formatting
  to the PR in case it hasn't been run locally prior to being committed.

Code Formatting
---------------
To avoid the tedium of meticulously formatting all the code ourselves, and to ensure as
standard style of formatting and sytactical idioms across the codebase, we use several
automatic code formatters, which run as pre-commit hooks. Many of them can also be
integrated direclty into your text editor or IDE with the appropriate plugins. The
following formatters are included in the template ``.pre-commit-config.yaml``:

* `Use only absolute import paths <https://github.com/MarcoGorelli/absolufy-imports>`__
* `Standardize the sorting of imports <https://github.com/PyCQA/isort>`__
* `Remove unneccesary f-strings <https://github.com/dannysepler/rm_unneeded_f_str>`__
* `Upgrade type hints for built-in types <https://github.com/sondrelg/pep585-upgrade>`__
* `Upgrade Python syntax <https://github.com/asottile/pyupgrade>`__
* `Deterministic formatting with Black <https://github.com/psf/black>`__
* We also have a custom hook that clears Jupyter notebook outputs prior to committing.

Code & Documentation Linters
----------------------------
To catch errors before commits are made, and to ensure uniform formatting across the
codebase, we also use a bunch of different linters. They don't change the code or
documentation files, but they will raise an error or warning when something doesn't
look right so you can fix it.

* `bandit <https://bandit.readthedocs.io/en/latest/>`__ identifies code patterns known
  to cause security issues.
* `doc8 <https://github.com/pycqa/doc8>`__ and `rstcheck
  <https://github.com/myint/rstcheck>`__ look for formatting issues in our docstrings
  and the standalone ReStructuredText (RST) files under the ``docs/`` directory.
* `flake8 <https://github.com/PyCQA/flake8>`__ is an extensible Python linting
  framework, with a bunch of plugins.
* `mypy <https://mypy.readthedocs.io/en/stable/index.html>`__ Does static type checking,
  and ensures that our code uses type annotations.
* `pre-commit <https://pre-commit.com>`__ has a collection of built-in checks that `use
  pygrep to search Python files <https://github.com/pre-commit/pygrep-hooks>`__ for
  common problems like blanket ``# noqa`` annotations, as well as `language agnostic
  problems <https://github.com/pre-commit/pre-commit-hooks>`__ like accidentally
  checking large binary files into the repository or having unresolved merge conflicts.
* `hadolint <https://github.com/AleksaC/hadolint-py>`__ checks Dockerfiles for errors
  and violations of best practices. It runs as a pre-commit hook.

Test Coverage
-------------
* We use Tox and a the pytest `coverage <https://coverage.readthedocs.io/en/6.3.2/>`__
  plugin to measure and record what percentage of our codebase is being tested, and to
  identify which modules, functions, and individual lines of code are not being
  exercised by the tests.
* When you run ``tox`` or ``tox -e ci`` (which is equivalent) a summary of the test
  coverage will be printed at the end of the tests (assuming they succeed). The full
  details of the test coverage is written to ``coverage.xml``.
* There are some configuration options for this process set in the ``.coveragerc`` file
  in the top level directory of the repository.
* When the tests are run via the ``tox-pytest`` workflow in GitHub Actions, the test
  coverage data from the ``coverage.xml`` output is uploaded to a service called
  `CodeCov <https://about.codecov.io/>`__ that saves historical data about our test
  coverage, and provides a nice visual representation of the data -- identifying which
  subpackages, modules, and individual lines of are being tested. For example, here are
  the results
  `for the cheshire repo <https://app.codecov.io/gh/catalyst-cooperative/cheshire>`__.
* The connection to CodeCov is configured in the ``.codecov.yml`` YAML file.
* In theory, we should be able to automatically turn CodeCov on for all of our GitHub
  repos, and it just Just Work, but in practice we've had to turn it on in the GitHub
  configuration one-by-one. Open source repositories are also supposed to be able to
  upload to the CodeCov site without requiring authentication, but this also hasn't
  worked, so thus far we've needed to request a new token for each repository. This
  token is stored in ``.codecov.yml``.
* Once it's enabled, CodeCov also adds a couple of test coverage checks to any pull
  request, to alert us if a PR reduces overall test coverage (which we would like to
  avoid).

Documentation Builds
--------------------
* We build our documentation using `Sphinx <https://www.sphinx-doc.org/en/master/>`__.
* Standalone docs files are stored under the ``docs/`` directory, and the Sphinx
  configuration is there in ``conf.py`` as well.
* We use `Sphinx AutoAPI <https://sphinx-autoapi.readthedocs.io/en/latest/>`__ to
  convert the docstrings embedded in the python modules under ``src/`` into additional
  documentation automatically.
* The top level documentation index simply includes this ``README.rst``, the
  ``LICENSE.txt`` and ``CODE_OF_CONDUCT.md`` files are similarly referenced. The only
  standalone documentation file under ``docs/`` right now is the ``release_notes.rst``.
* Unless you're debugging something specific, the docs should always be built using
  ``tox -e docs`` as that will lint the source files using ``doc8`` and ``rstcheck``,
  and wipe previously generated documentation to build everything from scratch. The docs
  are also rebuilt as part of the normal Tox run (equivalent to ``tox -e ci``).
* If you add something to the documentation generation process that needs to be cleaned
  up after, it should be integrated with the Sphinx hooks. There are some examples of
  how to do this at the bottom of ``docs/conf.py`` in the "custom build operations"
  section. For example, this is how we automatically regenerate the data dictionaries
  based on the PUDL metadata whenever the docs are built, ensuring that the docs stay up
  to date.

Documentation Publishing
------------------------
* We use the popular `Read the Docs <https://readthedocs.io>`__ service to host our
  documentation.
* When you open a PR, push to ``dev`` or ``main``, or tag a release, the associated
  documentation is automatically built on Read the Docs.
* There's some minimal configuration stored in the ``.readthedocs.yml`` file, but
  setting up this integration for a new repository requires some setup on the Read the
  Docs site.
* Create an account on Read the Docs using your GitHub identity, go to "My Projects"
  under the dropdown menu in the upper righthand corner, and click on "Import a
  Project." It should list the repositories that you have access to on GitHub. You may
  need to click on the Catalyst Cooperative logo in the right hand sidebar.
* It will ask you for a project name -- this will become part of the domain name for the
  documentation page on RTD and should be the same as the distribution name, but with
  dots and underscores replaced with dashes. E.g. ``catalystcoop-cheshire`` or
  ``catalystcoop-pudl-catalog``.
* Under Advanced Settings, make sure you
  `enable builds on PRs <https://docs.readthedocs.io/en/stable/pull-requests.html>`__.
  This will add a check ensuring that the documentation has built successfully on RTD
  for any PR in the repo.
* Under the Builds section for the new project (repo) you'll need to tell it which
  branches you want it to build, beyond the default ``main`` branch.
* Once the repository is connected to Read the Docs, an initial build of the
  documentation from the ``main`` branch should start.

Dependabot
----------
We use GitHub's `Dependabot <https://docs.github.com/en/code-security/dependabot/dependabot-version-updates>`__
to automatically update the allowable versions of packages we depend on. This applies
to both the Python dependencies specified in ``setup.py`` and to the versions of the
`GitHub Actions <https://docs.github.com/en/actions>`__ that we employ. The dependabot
behavior is configured in ``.github/dependabot.yml``

GitHub Actions
--------------
Under ``.github/workflows`` are YAML files that configure the `GitHub Actions
<https://docs.github.com/en/actions>`__ associated with the repository. We use GitHub
Actions to:

* Run continuous integration using `tox <https://tox.wiki>`__ on several different
  versions of Python.
* Build a Docker container with `repo2docker <https://github.com/marketplace/actions/repo2docker-action>`__
  which encapsulates the conda environment defined by the top level ``environment.yml``
  Note that for this action to succeed, you will need to
  `create a personal access token on Docker Hub <https://docs.docker.com/docker-hub/access-tokens/>`__
  and create new repository secrets to store your username and token called
  ``DOCKERHUB_USERNAME`` and ``DOCKERHUB_TOKEN`` and make sure that the Docker Hub
  repository you're trying to push to exists.
* Build a Docker container directly and push it to Docker Hub using the
  `docker-build-push action <https://github.com/docker/build-push-action>`__.

About Catalyst Cooperative
=======================================================================================
`Catalyst Cooperative <https://catalyst.coop>`__ is a small group of data
wranglers and policy wonks organized as a worker-owned cooperative consultancy.
Our goal is a more just, livable, and sustainable world. We integrate public
data and perform custom analyses to inform public policy (`Hire us!
<https://catalyst.coop/hire-catalyst>`__). Our focus is primarily on mitigating
climate change and improving electric utility regulation in the United States.

Contact Us
----------
* For general support, questions, or other conversations around the project
  that might be of interest to others, check out the
  `GitHub Discussions <https://github.com/catalyst-cooperative/pudl/discussions>`__
* If you'd like to get occasional updates about our projects
  `sign up for our email list <https://catalyst.coop/updates/>`__.
* Want to schedule a time to chat with us one-on-one? Join us for
  `Office Hours <https://calend.ly/catalyst-cooperative/pudl-office-hours>`__
* Follow us on Twitter: `@CatalystCoop <https://twitter.com/CatalystCoop>`__
* More info on our website: https://catalyst.coop
* For private communication about the project or to hire us to provide customized data
  extraction and analysis, you can email the maintainers:
  `pudl@catalyst.coop <mailto:pudl@catalyst.coop>`__
