from torchsr.train.options import args

import logging
# import tqdm

# class TqdmLoggingHandler(logging.StreamHandler):
# # class TqdmLoggingHandler(logging.Handler):
#     # def __init__(self, level=logging.NOTSET):
#     #     super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record) 

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO)

formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> \n %(message)s')

streamHandler = logging.StreamHandler()
# streamHandler.setStream(tqdm)
# streamHandler = TqdmLoggingHandler()
streamHandler.setFormatter(formatter)

if args.log is not None:
    import os
    # if not os.path.exists(args.log_dir):
    #     os.makedirs(args.log_dir)

    fileHandler = logging.FileHandler(args.log)
else:
    import os
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    fileHandler = logging.FileHandler(args.log_dir + '/log.txt')

fileHandler.setFormatter(formatter)



LOG.addHandler(streamHandler)
LOG.addHandler(fileHandler)

