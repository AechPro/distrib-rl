from . import (
    Message,
    communication_exception_handler,
)

import win32file
import win32pipe


class CommunicationHandler(object):
    NOVELTY_MAZE_GLOBAL_PIPE_ID = "NOVELTY_MAZE_GLOBAL_COMM_PIPE"
    NOVELTY_MAZE_GLOBAL_PIPE_NAME = r"\\.\pipe\NOVELTY_MAZE_GLOBAL_COMM_PIPE"
    NOVELTY_MAZE_DEFAULT_PIPE_SIZE = 4096

    def __init__(self):
        self._current_pipe_name = CommunicationHandler.NOVELTY_MAZE_GLOBAL_PIPE_NAME
        self._pipe = None
        self._connected = False

    def receive_message(self, header=None, num_attempts=100):
        # TODO: deal with discarded messages while waiting for a specific header
        if not self.is_connected():
            print("NOVELTY MAZE ATTEMPTED TO RECEIVE MESSAGE WITH NO CONNECTION")
            return communication_exception_handler.BROKEN_PIPE_ERROR

        m = Message()
        received_message = Message()
        exception_code = None
        for i in range(num_attempts):
            try:
                code, msg_bytes = win32file.ReadFile(
                    self._pipe, CommunicationHandler.NOVELTY_MAZE_DEFAULT_PIPE_SIZE
                )

            # This is the pywintypes.error object type.
            except BaseException as e:
                exception_code = communication_exception_handler.handle_exception(e)
                break

            msg_str = bytes.decode(msg_bytes)
            m.deserialize(msg_str)
            # print("COMM HANDLER GOT MESSAGE",m.header)

            # Only deserialize valid messages.
            if header is None or header == m.header:
                received_message.deserialize(msg_str)
                # Peek the next message in the pipe to see if we've reached the end of new messages.
                data = win32pipe.PeekNamedPipe(
                    self._pipe, CommunicationHandler.NOVELTY_MAZE_DEFAULT_PIPE_SIZE
                )
                if data[0] == b"":
                    break

        # TODO: make sure users of this object deal with the null message response
        return received_message, exception_code

    def send_message(self, message=None, header=None, body=None):
        if not self.is_connected():
            print("NOVELTY MAZE ATTEMPTED TO SEND MESSAGE WITH NO CONNECTION")
            return communication_exception_handler.BROKEN_PIPE_ERROR

        if message is None:
            if header is None:
                header = Message.NOVELTY_MAZE_NULL_MESSAGE_HEADER

            if body is None:
                body = Message.NOVELTY_MAZE_NULL_MESSAGE_BODY

            message = Message(header=header, body=body)

        serialized = message.serialize()
        exception_code = None
        try:
            win32file.WriteFile(self._pipe, str.encode(serialized))

        except BaseException as e:
            exception_code = communication_exception_handler.handle_exception(e)

        return exception_code

    def open_pipe(self, pipe_name=None, num_allowed_instances=1):
        if pipe_name is None:
            pipe_name = CommunicationHandler.NOVELTY_MAZE_GLOBAL_PIPE_NAME

        if self.is_connected():
            self.close_pipe()

        self._connected = False

        self._pipe = win32pipe.CreateNamedPipe(
            pipe_name,
            win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
            win32pipe.PIPE_TYPE_MESSAGE
            | win32pipe.PIPE_READMODE_MESSAGE
            | win32pipe.PIPE_WAIT,
            num_allowed_instances,
            CommunicationHandler.NOVELTY_MAZE_DEFAULT_PIPE_SIZE,
            CommunicationHandler.NOVELTY_MAZE_DEFAULT_PIPE_SIZE,
            0,
            None,
        )

        win32pipe.ConnectNamedPipe(self._pipe)

        self._current_pipe_name = pipe_name
        self._connected = True

    def close_pipe(self):
        self._connected = False
        win32file.CloseHandle(self._pipe)

    def is_connected(self):
        return self._connected

    @staticmethod
    def format_pipe_id(pipe_id):
        return r"\\.\pipe\{}".format(pipe_id)
