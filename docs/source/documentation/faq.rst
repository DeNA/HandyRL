==========
FAQ
==========

* How to use rule-based AI as an opponent?
    * You can easily use it by creating a rule-based AI method :code:`rule_based_action()` in a class :code:`Environment`.
* How to change the opponent in evaluation?
    * Set your agent in :code:`evaluation.py` like :code:`agents = [agent1, YourOpponentAgent()]`
* "*Too many open files*" Error
    * This error happens in a large-scale training. You should increase the maximum file limit by running :code:`ulimit -n 65536`. The value 65536 depends on a training setting. Note that the effect of :code:`ulimit` is session-based so you will have to either change the limit permanently (OS and version dependent) or run this command in your shell starting script.
    * In Mac OSX, you may need to change the system limit with :code:`launchctl` before running :code:`ulimit -n` (e.g. [How to Change Open Files Limit on OS X and macOS](https://gist.github.com/tombigel/d503800a282fcadbee14b537735d202c))

