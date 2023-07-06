import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import *
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
import logging
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_layers import *

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=500,
    message_steps=4,
    num_attention_heads=8,
    dense_units=2048,
    activ = 'sigmoid'
):
    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    
    atom_partition_indices = layers.Input(
        (), dtype="int32", name="atom_partition_indices"
    )

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = PartitionPadding(batch_size)([x, atom_partition_indices])

    x = layers.Masking()(x)

    x = TransformerEncoder(num_attention_heads, message_units, dense_units)(x)

    x = layers.GlobalAveragePooling1D()(x)
    #x=layers.Dropout(0.2)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x=layers.Dropout(0.2)(x)
    x = layers.Dense(dense_units/2, activation="relu")(x)
    x=layers.Dropout(0.2)(x)
    x = layers.Dense(560, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, atom_partition_indices],
        outputs=[x],
    )
    return model