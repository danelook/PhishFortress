from flask import Flask, request, render_template, jsonify
import joblib
import os
import csv
from cryptography.fernet import Fernet
