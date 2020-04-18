from flask import Flask, request


import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sklearn as sk
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

from app import views
from app import admin_views
