"""Style model training: profiles, LLaMA-Factory configuration, the encrypted training package,
the AutoDL control over ``asyncssh`` and the model registry (round 13).

``lf_template``
    the format strings of the ``qwen3_nothink`` template (the shared source of the prompt format);
``profiles``, ``yaml_render``, ``versions``, ``layout``
    the four GPU profiles, the YAML files rendered for them and the pinned software versions;
``dataset_dir``, ``bundle``, ``bundle_crypto``
    the exported dataset directory and the encrypted package built from it;
``remote``
    login, transfers, background jobs and the steps of ``twin train remote``;
``runs``, ``registry``
    the records of the runs (``training_runs``) and the registered models (``model_registry``);
``cli``, ``model_cli``
    the ``twin train`` and ``twin model`` commands.

The scripts that run on the instance and the YAML templates are in ``training/`` at the root of
the repository.
"""
