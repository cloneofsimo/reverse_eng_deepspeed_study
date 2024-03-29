# '''
# copied from https://github.com/facebookresearch/fairseq/blob/main/fairseq/pdb.py
# this is for multinode debugging (in terminal)
# '''
# import multiprocessing
# import os
# import pdb
# import sys


# __all__ = ["set_trace"]

# _stdin = [None]
# _stdin_lock = multiprocessing.Lock()
# try:
#     _stdin_fd = sys.stdin.fileno()
# except Exception:
#     _stdin_fd = None

# class MultiprocessingPdb(pdb.Pdb):
#     '''
#     cd /path/to/dir/multiprocessing_pdb &&\
#     pip install -e .
#     from multiprocessing_pdb import set_trace as Tra
#     Tra()
#     '''

#     def __init__(self):
#         pdb.Pdb.__init__(self, nosigint=True)

#     def _cmdloop(self):
#         stdin_bak = sys.stdin
#         with _stdin_lock:
#             try:
#                 if _stdin_fd is not None:
#                     if not _stdin[0]:
#                         _stdin[0] = os.fdopen(_stdin_fd)
#                     sys.stdin = _stdin[0]
#                 self.cmdloop()
#             finally:
#                 sys.stdin = stdin_bak

#     # def _cmdloop(self):
#     #     stdin_bak = sys.stdin
#     #     with _stdin_lock:
#     #         try:
#     #             # Redirect stdin only if we successfully captured its file descriptor
#     #             if _stdin_fd is not None and not _stdin[0]:
#     #                 _stdin[0] = os.fdopen(_stdin_fd)
#     #             sys.stdin = _stdin[0] if _stdin[0] else stdin_bak
#     #             self.cmdloop()
#     #         finally:
#     #             # Restore the original stdin to its rightful place
#     #             sys.stdin = stdin_bak

# def set_trace():
#     pdb = MultiprocessingPdb()
#     pdb.set_trace(sys._getframe().f_back)



'''
copied from https://github.com/stas00/ml-engineering/blob/master/debug/pytorch.md#invoke-pdb-on-a-specific-rank-in-multi-node-training
'''
import sys
import pdb


class MultiprocessingPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin