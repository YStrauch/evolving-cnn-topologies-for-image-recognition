from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import os
import webbrowser


class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


def serve():
    if os.path.isdir('visualise'):
        os.chdir('visualise')

    if not os.path.isfile('server.pem'):
        os.system("openssl req -new -keyout server.pem -out server.pem -x509 -days 365 -nodes -subj '/CN=www.pubnub.com/O=PubNub/C=US'")

    os.chdir('frontend')
    httpd = HTTPServer(('0.0.0.0', 1234), CORSRequestHandler)
    httpd.socket = ssl.wrap_socket(
        httpd.socket, certfile='../server.pem', server_side=True)
    webbrowser.open("https://0.0.0.0:1234")
    print("Serving at https://0.0.0.0:1234")

    httpd.serve_forever()


if __name__ == "__main__":
    serve()
