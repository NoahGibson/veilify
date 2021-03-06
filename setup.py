"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup

APP = ['main.py']
APP_NAME = 'Veilify'
DATA_FILES = [('', ['overlays'])]
OPTIONS = dict(
    plist=dict(
        NSCameraUsageDescription='Veilify requires access to the camera.'
    )
)

setup(
    app=APP,
    name=APP_NAME,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
