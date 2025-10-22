"""Filesystem middleware for providing file tools to agents."""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from deepagents.state import FilesystemState
from deepagents.tools import ls, read_file, write_file, edit_file

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`

You have access to a local, private filesystem which you can interact with using these tools.
- ls: list all files in the local filesystem
- read_file: read a file from the local filesystem
- write_file: write to a file in the local filesystem
- edit_file: edit a file in the local filesystem"""


class FilesystemMiddleware(AgentMiddleware):
    """Middleware that provides filesystem tools to agents."""

    state_schema = FilesystemState
    tools = [ls, read_file, write_file, edit_file]

    def modify_model_request(self, request: ModelRequest, agent_state: FilesystemState) -> ModelRequest:
        """Add filesystem system prompt to the model request."""
        request.system_prompt = request.system_prompt + "\n\n" + FILESYSTEM_SYSTEM_PROMPT
        return request
