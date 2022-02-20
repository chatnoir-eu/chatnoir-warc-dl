from queue import Queue

from pyspark import TaskContext, AccumulatorParam


class ResultsParam(AccumulatorParam):
    def __init__(self, q):
        self.q = q
        super().__init__()

    def zero(self, v):
        return []

    def addInPlace(self, acc1, acc2):
        # This is executed on the workers so we have to
        # merge the results
        if (TaskContext.get() is not None and
                TaskContext().get().partitionId() is not None):
            acc1.extend(acc2)
            return acc1
        else:
            # This is executed on the driver so we discard the results
            # and publish to self instead
            assert len(acc1) == 0
            for x in acc2:
                self.q.put(x)
            return []


class NonPicklableQueue(Queue):
    def __getstate__(self):
        return None
