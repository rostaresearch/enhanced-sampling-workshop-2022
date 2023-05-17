# from builtins import int, open, len, range
import os
import numpy as np
import mdtraj as md

import read_pdb as pdb
from src.string import opt_nw, opt_string

class UnboundException(Exception):
    pass


class Cluster:
    def __init__(self):
        self.atoms = []
        self.contacts = 0
        self.xyz = []
        self.masses = []
        self.com = None

    def addAtom(self, index):
        """
        this takes only the atom index, although atom types or protein residue names could also be saved
        :param index:
        :return:
        """
        self.atoms.append(index)

    def hasAtom(self, index):
        if index in self.atoms:
            return True
        else:
            return False

    def getCoM(self):
        massWeighted = []
        for i in range(len(self.atoms)):
            massWeighted.append(self.xyz[i] * self.masses[i])
        return np.sum(massWeighted, axis=0) / np.sum(self.masses)

    def getCoM_new(self, traj):
        return md.compute_center_of_mass(traj.atom_slice(self.atoms)) * 10


class Contact:
    def __init__(self):
        self.ligandClusters = []
        self.proteinClusters = []
        self.associations = []
        self.weight = []
        self.pdb = pdb.PDB()
        self.traj = 0
        self.step = 0
        self.indices = []

    def getLigandCluster(self, index):
        for i in range(len(self.ligandClusters)):
            if self.ligandClusters[i].hasAtom(index):
                return i
        self.ligandClusters.append(Cluster())
        self.ligandClusters[-1].addAtom(index)
        return len(self.ligandClusters) - 1

    def getProteinCluster(self, index):
        for i in range(len(self.proteinClusters)):
            if self.proteinClusters[i].hasAtom(index):
                return i
        self.proteinClusters.append(Cluster())
        self.proteinClusters[-1].addAtom(index)
        return len(self.proteinClusters) - 1

    def readClusters(self, filename, COM=True):
        """

        :param filename: list of selected distances with distances displayed
        :param COM: use centre of mass
        :return:
        """
        if COM:
            NEWCONTACT = True
            with open(filename, 'r') as f:
                for line in f:
                    if len(line.split()) == 1:
                        NEWCONTACT = True
                        w = 1.0
                    elif len(line.split()) == 2:
                        NEWCONTACT = True
                        try:
                            w = float(line.split()[1])
                        except ValueError:
                            w = 1.0
                            print("Warning, your coefficient is not a float. Check!")
                            print(line)
                    elif NEWCONTACT:
                        NEWCONTACT = False
                        l = self.getLigandCluster(int(line.split()[0]))
                        p = self.getProteinCluster(int(line.split()[3]))
                        self.ligandClusters[l].contacts += 1
                        self.proteinClusters[p].contacts += 1
                        self.associations.append((l, p))
                        self.weight.append(w)
                    else:
                        if not self.ligandClusters[l].hasAtom(int(line.split()[0])):
                            self.ligandClusters[l].addAtom(int(line.split()[0]))
                        if not self.proteinClusters[p].hasAtom(int(line.split()[3])):
                            self.proteinClusters[p].addAtom(int(line.split()[3]))
        else:
            with open(filename, 'r') as f:
                for line in f:
                    if not len(line.split()) == 1:
                        l = self.getLigandCluster(int(line.split()[0]))
                        p = self.getProteinCluster(int(line.split()[3]))
                        self.ligandClusters[l].contacts += 1
                        self.proteinClusters[p].contacts += 1
                        self.associations.append((l, p))
        return

    def getIndices(self):
        for c in self.ligandClusters:
            for i in c.atoms:
                if i not in self.indices:
                    self.indices.append(i)
        for c in self.proteinClusters:
            for i in c.atoms:
                if i not in self.indices:
                    self.indices.append(i)
        self.indices = np.sort(self.indices)
        return

    def getSumOfDistances(self):
        """
        :return:
        """
        self.getIndices()
        for c in self.ligandClusters:
            for a in c.atoms:
                c.masses.append(self.pdb.structure.topology._atoms[a].element.mass)
                c.xyz.append(self.pdb.structure._xyz[-1][a] * 10)
        for c in self.proteinClusters:
            for a in c.atoms:
                c.masses.append(self.pdb.structure.topology._atoms[a].element.mass)
                c.xyz.append(self.pdb.structure._xyz[-1][a] * 10)
        sum = 0.0
        for i in range(len(self.associations)):
            sum += self.weight[i]*(np.linalg.norm(self.ligandClusters[self.associations[i][0]].getCoM() - self.proteinClusters[self.associations[i][1]].getCoM()))
        return sum

    def prepareString(self, Unb, force=20):
        if os.path.isfile(os.path.join(Unb.wrkdir, "distances.csv")):
            distances = np.genfromtxt(os.path.join(Unb.wrkdir, "distances.csv"), delimiter=',')
        else:
            for a in range(1, Unb.cycle):
                # indices gives an error with small fragments
                # self.pdb.readDCD("traj_{0:d}/traj_{0:d}.dcd".format(a), Unb.top, indices=self.indices)
                temptraj = md.load("traj_{0:d}/traj_{0:d}.dcd".format(a), top=Unb.top)
                # temptraj = temptraj.image_molecules(inplace=True)
                for c in self.ligandClusters:
                    if c.com is None:
                        c.com = c.getCoM_new(temptraj)
                    else:
                        c.com = np.concatenate([c.com, c.getCoM_new(temptraj)])
                for c in self.proteinClusters:
                    if c.com is None:
                        c.com = c.getCoM_new(temptraj)
                    else:
                        c.com = np.concatenate([c.com, c.getCoM_new(temptraj)])
            distances = []
            for i in range(len(self.associations)):
                distances.append(np.linalg.norm(self.ligandClusters[self.associations[i][0]].com -
                                                self.proteinClusters[self.associations[i][1]].com, axis=1))
            distances = np.transpose(np.array(distances))
            np.savetxt(os.path.join(Unb.wrkdir, "distances.csv"), distances, fmt="%.3f", delimiter=',')
        # Writing the actual colvar file
        # nc = opt_nw(distances)
        # np.savetxt("pyconstr.dat", np.transpose(nc), fmt="%8.4f")
        nc = opt_string(distances.reshape((11, 5000, 5)), new_nw=50, write=True)
        steps = Unb.traj_length * 500
        current = np.empty(nc.shape)
        for i in range(nc.shape[0]):
            frame_similarity = np.linalg.norm(distances - nc[i], axis=1)
            index = np.where(frame_similarity == np.min(frame_similarity))[0][0]
            current[i] = distances[index]
            if np.linalg.norm(distances[index] - nc[i]) > 3:
                print("WARNING: the colvar positions are far from the path in window {:d}".format(i + 1))
                print("Is the ligand already unbound?")
            traj = int(index / steps)
            frame = index % steps
            structure = md.load("traj_{0:d}/traj_{0:d}.dcd".format(traj + 1), top=Unb.top, frame=frame)
            structure.save_pdb("{0:d}/string_window{0:d}.pdb".format(i + 1))
            with open("{0:d}/string_window{0:d}.xsc".format(i + 1), "w") as f:
                f.write("# NAMD extended system configuration output file\n")
                f.write("#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z s_x s_y s_z s_u s_v s_w\n")
                f.write("{0:d} ".format(index))
                f.write("{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f} ".format(
                    *list(10 * structure.unitcell_vectors.flatten())))
                f.write("0 0 0 0 0 0 0 0 0\n")
        header = """#Collective variables
#generated by unbinding/src/contact.py 
Colvarsrestartfrequency 5000 
"""
        dist = """
colvar {{
  name V{0:d}
  distance {{
    componentCoeff {3:.1f}
    group1 {{atomnumbers {1:s}}}
    group2 {{atomnumbers {2:s}}}
  }}
}}
"""
        bias = """
harmonic {{ 
  colvars V{0:d} 
centers {1:8.4f} 
forceConstant {2:d}
}}
"""
        for w in range(nc.shape[0]):
            with open("{0:d}/string_{0:d}.col".format(w + 1), 'w') as f:
                f.write(header)
                k = 1
                for i in range(len(self.associations)):
                    lig = ""
                    pro = ""
                    for a in self.ligandClusters[self.associations[i][0]].atoms:
                        lig += "{:d} ".format(a+1)
                    for a in self.proteinClusters[self.associations[i][1]].atoms:
                        pro += "{:d} ".format(a+1)
                    f.write(dist.format(k, lig, pro, self.weight[i]))
                    f.write(bias.format(k, nc[w, i], force))
                    k += 1
        return

    def writeNAMDcolvar(self, filename, traj_length=10, step=0.1, force=20):
        """Traj_length is the number of ns for the trajectory
           step is the ratio, 0.1 A per distance per nanosecond"""
        header = """#Collective variables
#generated by CoM_colvar.py 
Colvarstrajfrequency    1
Colvarsrestartfrequency 5000
colvar {
  name sum_1 
"""
        sum = self.getSumOfDistances()
        dist = """
  distance {{
    componentCoeff {2:.1f}
    group1 {{atomnumbers {0:s}}}
    group2 {{atomnumbers {1:s}}}
  }}"""
        footer = """
}}

harmonic {{
  colvars sum_1 
  centers {0:.2f}
  targetCenters {1:.2f}
  targetNumSteps {2:d}
  forceConstant {3:d}
}}""".format(sum, sum + (np.sum(self.weight) * traj_length * step), int(traj_length * 5E5), force)
        with open(filename, 'w') as f:
            f.write(header)
            if len(self.associations) == 0:
                raise UnboundException
            for i in range(len(self.associations)):
                lig = ""
                pro = ""
                for a in self.ligandClusters[self.associations[i][0]].atoms:
                    lig += "{:d} ".format(a+1)
                for a in self.proteinClusters[self.associations[i][1]].atoms:
                    pro += "{:d} ".format(a+1)
                f.write(dist.format(lig, pro, self.weight[i]))
            f.write(footer)
        return


def main():
    c = Contact()
    c.readClusters("new-distances.dat")
    # c.writeNAMDcolvar("sum.col")
    print("Done")


if __name__ == "__main__": main()