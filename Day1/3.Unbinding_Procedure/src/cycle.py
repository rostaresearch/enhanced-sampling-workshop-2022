import os
import sys

import mdtraj as md
import numpy as np

from src import pair
from src import contact as cv
from src.read_pdb import DCDnotReadable


class Cycle:
    def __init__(self, unbinding):
        self.number = unbinding.cycle
        self.wrkdir = unbinding.wrkdir
        self.template = unbinding.template
        self.traj_length = unbinding.traj_length
        self.prevtraj = None
        self.pairs = []
        self.contact = None

    def readDCD(self, top):
        try:
            self.prevtraj = md.load("traj_{0:d}/traj_{0:d}.dcd".format(self.number - 1), top=top)
        except OSError:
            raise DCDnotReadable
        return

    def saveNewDCD(self, stride):
        self.prevtraj[range(0, self.prevtraj.n_frames, stride)].save_dcd(os.path.join("traj_{0:d}/traj_{0:d}-wrapped.dcd".format(self.number - 1)))
        return

    def getNeighbour(self, ligres, cutoff, ligandClusters):
        lig = self.prevtraj.top.select("resname {:s} and not type H".format(ligres))
        protein = self.prevtraj.top.select("protein and not type H")
        neighbours = []
        for l in lig:
            neighbours.append(md.compute_neighbors(self.prevtraj, cutoff=0.1*cutoff, haystack_indices=protein, query_indices=[l]))
        data = []
        for i in range(self.prevtraj.n_frames):
            frame = []
            for j in range(len(lig)):
                for pa in neighbours[j][i]:
                    frame.append([lig[j],
                                 self.prevtraj.top._atoms[lig[j]].residue.resSeq,
                                 self.prevtraj.top._atoms[lig[j]].name,
                                 pa,
                                 self.prevtraj.top._atoms[pa].residue.resSeq,
                                 self.prevtraj.top._atoms[pa].name,
                                 10 * np.linalg.norm(self.prevtraj._xyz[i, lig[j]] -
                                                     self.prevtraj._xyz[i, pa]),
                                 self.prevtraj.top._atoms[pa].residue.name,
                                 ligres])
            data.append(frame)
        self.pairs = pair.createPairs(data)
        for i in range(len(self.pairs)):
            self.pairs[i].getProteinClusterAtoms(self.prevtraj.top)
            self.pairs[i].getLigandClusterAtoms(self.prevtraj.top, ligandClusters)
        self.removeDuplicates()
        return

    def getClusters(self, ligandClusters):
        """Obsolete as has to be done earlier"""
        for i in range(len(self.pairs)):
            self.pairs[i].getProteinClusterAtoms(self.prevtraj.top)
            self.pairs[i].getLigandClusterAtoms(self.prevtraj.top, ligandClusters)
        self.removeDuplicates()
        return

    def removeDuplicates(self):
        remove = []
        for i in range(len(self.pairs)):
            for j in range(i):
                if self.pairs[j].hasAtom(self.pairs[i].atom["index"]) and self.pairs[j].hasAtom(self.pairs[i].ligand_atom["index"]):
                    union = sorted(list(set(self.pairs[i].mask) | set(self.pairs[j].mask)))
                    value = []
                    for k in union:
                        try:
                            iv = self.pairs[i].value[self.pairs[i].mask.index(k)]
                        except ValueError:
                            value.append(self.pairs[j].value[self.pairs[j].mask.index(k)])
                            continue
                        try:
                            jv = self.pairs[j].value[self.pairs[j].mask.index(k)]
                        except ValueError:
                            value.append(self.pairs[i].value[self.pairs[i].mask.index(k)])
                            continue
                        value.append(min([iv, jv]))
                    self.pairs[j].value = value
                    self.pairs[j].mask = union
                    self.pairs[j].count = len(self.pairs[j].mask)
                    remove.append(i)
        uniquepairs = []
        for i in range(len(self.pairs)):
            if i not in remove and self.pairs[i].count > (0.5 * self.prevtraj.n_frames):
                uniquepairs.append(self.pairs[i])
        self.pairs = uniquepairs
        return

    def getAllPairs(self, Unb):
        IDfound = []
        for c in Unb.pairs:
            for pair in c:
                if pair.ID not in IDfound:
                    self.pairs.append(pair)
                    IDfound.append(pair.ID)
        return

    def createContact(self, COM = True):
        self.contact = cv.Contact()
        self.contact.pdb.structure = self.prevtraj[-1]
        if COM:
            for pair in self.pairs:
                if len(pair.ligandClusterAtoms) == 0:
                    l = self.contact.getLigandCluster(pair.ligand_atom["index"])
                else:
                    l = self.contact.getLigandCluster(pair.ligandClusterAtoms[0]["index"])
                    for a in pair.ligandClusterAtoms[1:]:
                        if not self.contact.ligandClusters[l].hasAtom(a["index"]):
                            self.contact.ligandClusters[l].addAtom(a["index"])
                if len(pair.proteinClusterAtoms) == 0:
                    p = self.contact.getProteinCluster(pair.atom["index"])
                else:
                    p = self.contact.getProteinCluster(pair.proteinClusterAtoms[0]["index"])
                    for a in pair.proteinClusterAtoms[1:]:
                        if not self.contact.proteinClusters[p].hasAtom(a["index"]):
                            self.contact.proteinClusters[p].addAtom(a["index"])
                self.contact.ligandClusters[l].contacts += 1
                self.contact.proteinClusters[p].contacts += 1
                self.contact.associations.append((l, p))
                self.contact.weight.append(1.0)
        else:
            for pair in self.pairs:
                if len(pair.ligandClusterAtoms) == 0:
                    l = self.contact.getLigandCluster(pair.ligand_atom["index"])
                else:
                    for a in pair.ligandClusterAtoms:
                        l = self.contact.getLigandCluster(a["index"])
                if len(pair.proteinClusterAtoms) == 0:
                    p = self.contact.getProteinCluster(pair.atom["index"])
                else:
                    for a in pair.proteinClusterAtoms[1:]:
                        p = self.contact.getProteinCluster(a["index"])
            for lc in self.contact.ligandClusters:
                for pair in self.pairs:
                    for l in pair.ligandClusterAtoms:
                        if l["index"] == lc.atoms[0]:
                            if len(pair.proteinClusterAtoms) == 0:
                                lc.contacts += 1
                                self.contact.associations.append((self.contact.ligandClusters.index(lc), self.contact.getProteinCluster(pair.atom["index"])))
                            else:
                                lc.contacts += len(pair.proteinClusterAtoms)
                                for a in pair.proteinClusterAtoms:
                                    self.contact.associations.append((self.contact.ligandClusters.index(lc),
                                                                      self.contact.getProteinCluster(a["index"])))
            for pc in self.contact.proteinClusters: # count for protein clusters doesn't work
                for pair in self.pairs:
                    for p in pair.proteinClusterAtoms:
                        if p["index"] == pc.atoms[0]:
                            if len(pair.ligandClusterAtoms) == 0:
                                pc.contacts += 1
                            else:
                                pc.contacts += len(pair.ligandClusterAtoms)
        return

    def setupCycle(self):
        if len(self.contact.associations) == 0:
            raise cv.UnboundException
        try:
            os.mkdir(os.path.join(self.wrkdir, "traj_{:d}".format(self.number)))
        except OSError:
            print("WARNING: traj_{:d} already exists. Files might be overwritten in the folder.".format(self.number))
            print("WARNING: Rename the folder to back it up, or remove.")
            sys.exit(0)
        self.writeNamdInput()

    def writeNamdInput(self):
        input = self.template.format(self.number, self.number - 1, int(self.traj_length * 5E5))
        with open(os.path.join(self.wrkdir, "traj_{:d}".format(self.number), "traj_{:d}.inp".format(self.number)), "w") as f:
            f.write(input)
        return
