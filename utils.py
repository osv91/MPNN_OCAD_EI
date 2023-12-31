from itertools import groupby
import numpy as np
from rdkit import Chem
class OCAD_SDF_reader():
    def __init__(self, entry):
        self.entry = entry
        self.spec_shape = 560
        self.mw = self.mw()
        self.spectrum = self.spectrum()
        self.neutral_loss = self.neutral_loss()
        self.name=self.name()
        self.ri = self.ri()
        self.inchikey = self.inchikey()
        self.formula = self.formula()
        
        self.id = self.id_n()
        self.comments = self.comments()
        self.max_mz=self.max_mz()
        self.smiles = self.smiles()
        self.schedule = self.schedule()
        
    def mol(self):
        idx = self.entry.index('M  END\n')+1
        molblock = ''.join(self.entry[:idx])
        return Chem.MolFromMolBlock(molblock)
    def name(self):
        idx = self.entry.index('>  <NAME>\n')+1
        return self.entry[idx].rstrip()
    def ri(self):
        try:
            idx = self.entry.index('>  <RETENTION INDEX>\n')+1
            ri_string = self.entry[idx].rstrip().lstrip('RI - ')        
            return [int(x) for x in ri_string.split(',')]
        except:
            return None
    def inchikey(self):
        idx = self.entry.index('>  <INCHIKEY>\n')+1
        return self.entry[idx].rstrip()
    def formula(self):
        idx = self.entry.index('>  <FORMULA>\n')+1
        return self.entry[idx].rstrip()
    def mw(self):
        idx = self.entry.index('>  <MW>\n')+1
        return int(self.entry[idx].rstrip())
    def cas(self):
        idx = self.entry.index('>  <CASNO>\n')+1
        return self.entry[idx].rstrip()
    def id_n(self):
        idx = self.entry.index('>  <ID>\n')+1
        return int(self.entry[idx].rstrip())
    def comments(self):
        idx = self.entry.index('>  <COMMENT>\n')+1
        return self.entry[idx].rstrip()
    def spectrum(self):
        idx = self.entry.index('>  <MASS SPECTRAL PEAKS>\n')+1
        spectrum_string = ''.join(self.entry[idx:-1])
        spectrum_list = spectrum_string.split('\n')[:-1]
        spectrum_dict = {int(x.split(' ')[0]):int(x.split(' ')[1]) for x in spectrum_list}
        spec_array = np.zeros(self.spec_shape)
        for x in spectrum_dict.keys():
            spec_array[x]= spectrum_dict[x]
        return spec_array
    def neutral_loss(self):
        neutral_loss_array = np.zeros(self.spec_shape)
        peaks = np.nonzero(self.spectrum)[0]
        for peak in peaks:
            neutral_loss_array[self.mw-peak] = self.spectrum[peak]
        return neutral_loss_array
    def max_mz(self):
        return int(self.entry[-2].rstrip().split(' ')[0])
    def smiles(self):
        return Chem.MolToSmiles(self.mol(), canonical=True, isomericSmiles=False)
    def schedule(self):
        if self.comments.split(', ')[-2].startswith('RI'):
            return self.comments.split(', ')[-3]
        else:
            return self.comments.split(', ')[-2]
            
            
            
            
            



def get_sdf_entries(sdf_file):
    lines = open(sdf_file, "r").readlines()
    sdf = [list(group) for k, group in groupby(lines, lambda x: x == "$$$$\n") if not k]
    return sdf
    
    
    
import numpy as np
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
def inchi2smiles(inchi):
    try:
        return Chem.MolToSmiles(MolFromInchi(inchi))
    except:
        return('NA')

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    def aromatic(self, atom):
        return atom.GetIsAromatic()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si', 'As', 'Se'},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "aromatic":{True, False}
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loop. Notice, this also helps against some edge cases where the
        # last node has no edges. Alternatively, if no self-loops are used, for these
        # edge cases, zero-padding on the output of the edge network is needed.
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        atom_neighbors = atom.GetNeighbors()

        for neighbor in atom_neighbors:
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)


    return (np.array(atom_features_list), np.array(bond_features_list), np.array(pair_indices_list) )

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices. atom_partition_indices will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    atom_partition_indices = tf.repeat(molecule_indices, num_atoms)
    bond_partition_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(
        tf.gather(increment, bond_partition_indices), [(num_bonds[0], 0)]
    )
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, atom_partition_indices), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    #convert arrays to tensors
    X = (tf.ragged.constant(X[0], dtype=tf.float32),
   	 tf.ragged.constant(X[1], dtype=tf.float32),
    	 tf.ragged.constant(X[2], dtype=tf.int64))
    
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1)