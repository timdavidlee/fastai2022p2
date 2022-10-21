# notebook settings
c = get_config()
c.NotebookApp.ip = "0.0.0.0"
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = "/home/ml"
c.NotebookApp.port = 8888
c.NotebookApp.terminado_settings = {"shell_command": ["/bin/bash"]}


c.NotebookApp.password = "sha1:35696359a58d:66dcc6f78139d9614102fd71b72c8b667d77492f"
c.FileContentsManager.delete_to_trash = False