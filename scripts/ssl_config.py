"""
SSL Configuration for Company Network
This module must be imported at the top of any script that uses internet traffic.
"""
import os
import ssl
import certifi

def setup_ssl_certificates():
    """
    Configure SSL certificates for company network.
    Must be called before any network operations (model downloads, etc.)
    """
    # Company SSL certificate path
    ssl_cert_path = '/Users/dagmawi.wube/cacerts.txt'

    # # Set environment variables for various libraries
    # os.environ['SSL_CERT_FILE'] = ssl_cert_path
    # os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
    # os.environ['CURL_CA_BUNDLE'] = ssl_cert_path
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # # Verify certificate file exists
    # if not os.path.exists(ssl_cert_path):
    #     raise FileNotFoundError(
    #         f"SSL certificate file not found: {ssl_cert_path}\n"
    #         "Please ensure the company certificate file exists at this path."
    #     )

    # print(f"✓ SSL certificates configured: {ssl_cert_path}")

    # # For urllib and other libraries that use ssl module
    # # Note: Some libraries may not respect this, but we set it anyway
    # ssl._create_default_https_context = ssl._create_unverified_context

    return ssl_cert_path

# Auto-configure when module is imported
setup_ssl_certificates()
