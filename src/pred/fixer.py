from dataclasses import dataclass

import rdkit.Chem as Chem
from rdkit.Chem import rdChemReactions


@dataclass
class SMARTSReaction:
    """
    A class to represent a reaction from SMARTS pattern
    """
    smarts: str
    name: str
    reaction: rdChemReactions.ChemicalReaction = None

    def __post_init__(self):
        """
        Create a reaction from SMARTS pattern
        """
        self.reaction = rdChemReactions.ReactionFromSmarts(self.smarts)

    def __call__(self, mol: Chem.Mol):
        """
        Run reaction on a molecule
        """
        return self.reaction.RunReactants((mol,))

    def __repr__(self):
        return f'<SMARTSReaction {self.name}>'


class MolFixer:
    """
    A class to fix molecules based on SMARTS patterns
    """

    def __init__(self) -> None:
        self.fixes = [SMARTSReaction(
            '[#6:2]1-[#6:3](-[#6:1])=[#6:4]-[#6:5]=[#6:6]-[#6:7]=[#6:8]-1>>[#6:3]1(-[#6:1])=[#6:4]-[#6:5]=[#6:6]-[#6:7]=[#6:8]-1',
            'cycloheptatriene to benzene'),
            SMARTSReaction(
                '[#6:0]-[#6:1]1-[#6:2]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-1>>[#6:0]-[#6:1]-[#6:2]1:[#6]:[#6]:[#6]:[#6]:[#6]:1',
                "uncycle conjugated cyclopropane to benzene"),
            SMARTSReaction(
                f'[#6]1=[#6]2-[#6:2]=[#6:1][#6:6]=[#6:5]-[#6:4]=[#6:3]-1-2>>[#6:2]1=[#6:1]-[#6:6]=[#6:5]-[#6:4]=[#6:3]-1',
                'bicyclo[5,3,1] to benzene')]

    def fix(self, mol: Chem.Mol):
        """
        Fix a molecule by applying all the fixes
        Usage:
        >>> fixer = MolFixer()
        >>> mol = Chem.MolFromSmiles('C1C=CC=CC=1')
        >>> fixed_mol = fixer.fix(mol)
        """
        for fix in self.fixes:
            ps = fix(mol)
            if len(ps) > 0:
                mol = ps[0][0]
        return mol
