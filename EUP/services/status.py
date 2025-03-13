

from server import PromptServer

class StatusService():

    def updateNodeStatus(self, node, text, progress=None):
        if PromptServer.instance.client_id is None:
            return

        PromptServer.instance.send_sync("EUP/update_status", {
            "node": node,
            "progress": progress,
            "text": text
        }, PromptServer.instance.client_id)