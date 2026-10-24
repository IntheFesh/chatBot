"""Remote control of the AutoDL instance with ``asyncssh`` (R-TRN-010).

``connection``
    host key policy and login (password from the credential store, or a key file);
``transfer``
    uploads that resume and downloads that verify, both checked by sha256;
``jobs``
    long steps that keep running when the connection drops, and following their log;
``steps``
    the steps of ``twin train remote`` and their bookkeeping in ``training_runs``;
``cli``
    the commands.
"""
