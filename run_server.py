from server import Server

host_ip = ['169.254.31.226', '192.168.0.165','169.254.59.105'][2]
# Example usage
if __name__ == "__main__":
    server = Server(host_ip=host_ip, 
                    frame_width=640, 
                    frame_height=480,
                    source='camera',
                    #img_dir="../captured_pngs/", 
                    label_path=None)
    server.run()
