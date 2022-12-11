# from builtins import Exception
import os
import sys
import mdtraj as md

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PDBnotReadable(Exception):
    pass


class DCDnotReadable(Exception):
    pass


class PDB:
    def __init__(self):
        self.structure = None
        return

    def read(self, file):
        try:
            self.structure = md.load(file)
        except:
            raise PDBnotReadable

    def readDCD(self, dcd, pdb, indices=None):
        try:
            self.structure = md.load_dcd(dcd, top=pdb, atom_indices=indices)
            # self.structure = self.structure.image_molecules(inplace=True)
        except:
            raise DCDnotReadable


def main():
    example = PDB()
    example.read("examples/4fku.pdb")
    print("stop")


if __name__ == "__main__":
    main()
