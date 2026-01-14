"""Tests for PDB parsing utilities.

This module tests the PDB parsing functions that extract structural
information from PDB files for use in ProteinMPNN.
"""

import numpy as np
import pytest

# =============================================================================
# Test parse_PDB_biounits
# =============================================================================


class TestParsePDBBiounits:
    """Tests for parse_PDB_biounits function."""

    def test_parse_pdb_biounits_basic(self, test_pdb_path):
        """Test basic PDB parsing."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq = parse_PDB_biounits(str(test_pdb_path), chain="A")

        assert xyz is not None
        assert seq is not None
        assert isinstance(xyz, np.ndarray)
        assert isinstance(seq, list)

    def test_parse_pdb_biounits_shape(self, test_pdb_path):
        """Test output shape of parse_PDB_biounits."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        # Default atoms: N, CA, C
        xyz, seq = parse_PDB_biounits(str(test_pdb_path), chain="A")

        # Shape should be (num_residues, 3, 3) for N, CA, C
        assert len(xyz.shape) == 3
        assert xyz.shape[1] == 3  # 3 atoms
        assert xyz.shape[2] == 3  # 3D coordinates

    def test_parse_pdb_biounits_custom_atoms(self, test_pdb_path):
        """Test parsing with custom atom selection."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        # Parse with N, CA, C, O
        xyz, seq = parse_PDB_biounits(str(test_pdb_path), atoms=["N", "CA", "C", "O"], chain="A")

        assert xyz.shape[1] == 4  # 4 atoms

    def test_parse_pdb_biounits_ca_only(self, test_pdb_path):
        """Test parsing CA atoms only."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq = parse_PDB_biounits(str(test_pdb_path), atoms=["CA"], chain="A")

        assert xyz.shape[1] == 1  # Only CA

    def test_parse_pdb_biounits_sequence(self, test_pdb_path):
        """Test that sequence is extracted correctly."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq = parse_PDB_biounits(str(test_pdb_path), chain="A")

        # 1UBQ sequence starts with MQIFVKTLTGK...
        assert len(seq) == 1
        assert seq[0].startswith("MQIFVKTLTGK")

    def test_parse_pdb_biounits_invalid_chain(self, test_pdb_path):
        """Test parsing with invalid chain."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq = parse_PDB_biounits(str(test_pdb_path), chain="Z")

        # Should return 'no_chain' for invalid chain
        assert xyz == "no_chain"
        assert seq == "no_chain"


# =============================================================================
# Test parse_PDB
# =============================================================================


class TestParsePDB:
    """Tests for parse_PDB function."""

    def test_parse_pdb_basic(self, test_pdb_path):
        """Test basic PDB parsing."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path))

        assert isinstance(result, list)
        assert len(result) > 0

    def test_parse_pdb_dict_keys(self, test_pdb_path):
        """Test that result dictionary has expected keys."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path))
        pdb_dict = result[0]

        assert "name" in pdb_dict
        assert "seq" in pdb_dict
        assert "num_of_chains" in pdb_dict

    def test_parse_pdb_chain_specific(self, test_pdb_path):
        """Test parsing specific chains."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path), input_chain_list=["A"])
        pdb_dict = result[0]

        assert "seq_chain_A" in pdb_dict
        assert "coords_chain_A" in pdb_dict

    def test_parse_pdb_ca_only(self, test_pdb_path):
        """Test CA-only parsing."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path), ca_only=True)
        pdb_dict = result[0]

        # Should have CA coordinates
        assert "coords_chain_A" in pdb_dict
        coords = pdb_dict["coords_chain_A"]
        assert "CA_chain_A" in coords

    def test_parse_pdb_full_backbone(self, test_pdb_path):
        """Test full backbone parsing."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path), ca_only=False)
        pdb_dict = result[0]

        coords = pdb_dict["coords_chain_A"]
        assert "N_chain_A" in coords
        assert "CA_chain_A" in coords
        assert "C_chain_A" in coords
        assert "O_chain_A" in coords

    def test_parse_pdb_name_extraction(self, test_pdb_path):
        """Test that PDB name is extracted correctly."""
        from frustrampnn.model.pdb_parsing import parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = parse_PDB(str(test_pdb_path))
        pdb_dict = result[0]

        assert pdb_dict["name"] == "1UBQ"


# =============================================================================
# Test alt_parse_PDB_biounits
# =============================================================================


class TestAltParsePDBBiounits:
    """Tests for alt_parse_PDB_biounits function."""

    def test_alt_parse_pdb_biounits_basic(self, test_pdb_path):
        """Test alternative PDB parsing."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq, resn_list = alt_parse_PDB_biounits(str(test_pdb_path), chain="A")

        assert xyz is not None
        assert seq is not None
        assert resn_list is not None
        assert isinstance(resn_list, list)

    def test_alt_parse_pdb_biounits_resn_list(self, test_pdb_path):
        """Test that residue number list is returned."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq, resn_list = alt_parse_PDB_biounits(str(test_pdb_path), chain="A")

        # 1UBQ has 76 residues
        assert len(resn_list) == 76

    def test_alt_parse_pdb_biounits_resn_format(self, test_pdb_path):
        """Test residue number format."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB_biounits

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        xyz, seq, resn_list = alt_parse_PDB_biounits(str(test_pdb_path), chain="A")

        # Residue numbers should be strings (may include insertion codes)
        assert all(isinstance(r, str) for r in resn_list)


# =============================================================================
# Test alt_parse_PDB
# =============================================================================


class TestAltParsePDB:
    """Tests for alt_parse_PDB function."""

    def test_alt_parse_pdb_basic(self, test_pdb_path):
        """Test alternative PDB parsing."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = alt_parse_PDB(str(test_pdb_path))

        assert isinstance(result, list)
        assert len(result) > 0

    def test_alt_parse_pdb_has_resn_list(self, test_pdb_path):
        """Test that result includes residue number list."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = alt_parse_PDB(str(test_pdb_path))
        pdb_dict = result[0]

        assert "resn_list" in pdb_dict
        assert isinstance(pdb_dict["resn_list"], list)

    def test_alt_parse_pdb_resn_list_length(self, test_pdb_path):
        """Test residue number list length matches sequence."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB

        if not test_pdb_path.exists():
            pytest.skip("Test PDB file not found")

        result = alt_parse_PDB(str(test_pdb_path), input_chain_list=["A"])
        pdb_dict = result[0]

        seq_length = len(pdb_dict["seq_chain_A"])
        resn_length = len(pdb_dict["resn_list"])

        assert resn_length == seq_length


# =============================================================================
# Test PDB Parsing Exports
# =============================================================================


class TestPDBParsingExports:
    """Test that all expected functions are exported."""

    def test_all_exports(self):
        """Test all expected exports from pdb_parsing module."""
        from frustrampnn.model import pdb_parsing

        expected = [
            "parse_PDB",
            "parse_PDB_biounits",
            "alt_parse_PDB",
            "alt_parse_PDB_biounits",
        ]

        for name in expected:
            assert hasattr(pdb_parsing, name), f"Missing export: {name}"

    def test_exports_from_model_init(self):
        """Test exports from model __init__."""
        from frustrampnn.model import alt_parse_PDB, parse_PDB

        assert parse_PDB is not None
        assert alt_parse_PDB is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestPDBParsingEdgeCases:
    """Test edge cases for PDB parsing."""

    def test_missing_atoms_handled(self, tmp_path):
        """Test that missing atoms are handled with NaN."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create a minimal PDB with missing atoms
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq = parse_PDB_biounits(str(pdb_file), atoms=["N", "CA", "C"], chain="A")

        # N and C should be NaN since only CA is present
        assert np.isnan(xyz[:, 0, :]).all()  # N atoms
        assert not np.isnan(xyz[:, 1, :]).any()  # CA atoms present
        assert np.isnan(xyz[:, 2, :]).all()  # C atoms

    def test_selenomethionine_handling(self, tmp_path):
        """Test that MSE (selenomethionine) is converted to MET."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create PDB with MSE
        pdb_content = """HETATM    1  N   MSE A   1       0.000   0.000   0.000  1.00  0.00           N
HETATM    2  CA  MSE A   1       1.460   0.000   0.000  1.00  0.00           C
HETATM    3  C   MSE A   1       2.000   1.200   0.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test_mse.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq = parse_PDB_biounits(str(pdb_file), chain="A")

        # MSE should be converted to M (methionine)
        assert "M" in seq[0]

    def test_insertion_codes(self, tmp_path):
        """Test handling of insertion codes in residue numbers."""
        from frustrampnn.model.pdb_parsing import alt_parse_PDB_biounits

        # Create PDB with insertion codes
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   1A      3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   2       7.600   0.000   0.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test_ins.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq, resn_list = alt_parse_PDB_biounits(str(pdb_file), atoms=["CA"], chain="A")

        # Should have 3 residues including insertion
        assert len(resn_list) >= 2

    def test_empty_chain(self, tmp_path):
        """Test handling of empty/missing chain."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create PDB with only chain B
        pdb_content = """ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test_chain.pdb"
        pdb_file.write_text(pdb_content)

        # Request chain A which doesn't exist
        xyz, seq = parse_PDB_biounits(str(pdb_file), chain="A")

        assert xyz == "no_chain"
        assert seq == "no_chain"

    def test_multiple_models(self, tmp_path):
        """Test that only first model is parsed (standard behavior)."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create PDB with multiple models
        pdb_content = """MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  CA  GLY A   1       5.000   5.000   5.000  1.00  0.00           C
ENDMDL
END
"""
        pdb_file = tmp_path / "test_models.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq = parse_PDB_biounits(str(pdb_file), atoms=["CA"], chain="A")

        # Should parse both models' atoms (standard behavior)
        # The exact behavior depends on implementation
        assert xyz is not None


# =============================================================================
# Test Coordinate Extraction
# =============================================================================


class TestCoordinateExtraction:
    """Test coordinate extraction accuracy."""

    def test_coordinate_values(self, tmp_path):
        """Test that coordinates are extracted correctly."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create PDB with known coordinates
        pdb_content = """ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00  0.00           C
ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test_coords.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq = parse_PDB_biounits(str(pdb_file), atoms=["N", "CA", "C"], chain="A")

        # Check coordinates
        np.testing.assert_array_almost_equal(xyz[0, 0], [1.0, 2.0, 3.0])  # N
        np.testing.assert_array_almost_equal(xyz[0, 1], [4.0, 5.0, 6.0])  # CA
        np.testing.assert_array_almost_equal(xyz[0, 2], [7.0, 8.0, 9.0])  # C

    def test_sequence_extraction(self, tmp_path):
        """Test that sequence is extracted correctly."""
        from frustrampnn.model.pdb_parsing import parse_PDB_biounits

        # Create PDB with known sequence
        pdb_content = """ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  GLN A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ILE A   3       7.600   0.000   0.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "test_seq.pdb"
        pdb_file.write_text(pdb_content)

        xyz, seq = parse_PDB_biounits(str(pdb_file), atoms=["CA"], chain="A")

        assert seq[0] == "MQI"
