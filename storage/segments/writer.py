import os
import struct
import json
import zlib
import time
import uuid
import numpy as np
from typing import Tuple

class BinarySegmentSerializer:
    """
    Handles binary serialization and deserialization of database segments (segment.bin).
    Format conforms strictly to the SYSTEM_ARCHITECTURE specifications:
    - 64-Byte Header
    - 32-Byte index records (aligned)
    - Padded vector blocks (aligned to 64-byte boundaries for SIMD/AVX vector calculations)
    - Serialized length-prefixed JSON payloads
    """

    HEADER_FORMAT = "<4s H 16s I I d Q 18x" # 64 Bytes
    # 4s  - Magic Number (b'VSEG')
    # H   - Version (1)
    # 16s - Segment UUID bytes
    # I   - Record Count (uint32)
    # I   - Deleted Count (uint32)
    # d   - Created Timestamp (double)
    # Q   - CRC32 Checksum (uint64)
    # 18x - 18 Padding bytes

    INDEX_RECORD_FORMAT = "<16s I I I B 3x" # 32 Bytes
    # 16s - Record UUID bytes
    # I   - Vector Absolute Offset (uint32)
    # I   - Payload Absolute Offset (uint32)
    # I   - Payload Length (uint32)
    # B   - Status (uint8: 1=Active, 2=Tombstone)
    # 3x  - 3 Padding bytes

    @classmethod
    def serialize(cls, segment_id: uuid.UUID, records: list[dict], dimension: int) -> bytes:
        """
        Serialize a list of vector records into segment binary bytes.
        Each record should contain:
        - 'id': uuid.UUID
        - 'vector': list of floats
        - 'payload': dict (JSON serializable)
        """
        record_count = len(records)
        deleted_count = sum(1 for r in records if r.get('status') == 2)
        created_ts = time.time()
        
        # Calculate alignment sizes
        header_size = 64
        index_table_size = record_count * 32
        
        # Vector block must start at a 64-byte boundary
        vector_block_start = ((header_size + index_table_size) + 63) // 64 * 64
        vector_padding_size = vector_block_start - (header_size + index_table_size)
        
        # Calculate vector padded sizing (64-byte alignment per vector)
        single_vector_raw_size = dimension * 4 # float32 = 4 bytes
        single_vector_padded_size = (single_vector_raw_size + 63) // 64 * 64
        
        # We assemble the body parts first to resolve absolute offsets
        vector_bytes_list = []
        payload_bytes_list = []
        
        # Map record positions
        index_entries = []
        
        current_vector_offset = vector_block_start
        # Compute payload block start (immediately following vector block)
        payload_block_start = vector_block_start + (record_count * single_vector_padded_size)
        current_payload_offset = payload_block_start
        
        for record in records:
            rec_id = record['id']
            vector = record['vector']
            payload = record.get('payload', {})
            status = record.get('status', 1) # 1=Active, 2=Tombstone
            
            # 1. Format Vector (float32 array with trailing padding to satisfy 64B alignment)
            v_arr = np.array(vector, dtype=np.float32)
            raw_v_bytes = v_arr.tobytes()
            padding_bytes_needed = single_vector_padded_size - len(raw_v_bytes)
            padded_v_bytes = raw_v_bytes + (b'\x00' * padding_bytes_needed)
            vector_bytes_list.append(padded_v_bytes)
            
            # 2. Format Payload (Length-prefixed JSON)
            payload_str = json.dumps(payload)
            payload_encoded = payload_str.encode('utf-8')
            payload_len = len(payload_encoded)
            # Prefix payload with its length (4-byte uint32)
            payload_block = struct.pack(f"<I {payload_len}s", payload_len, payload_encoded)
            payload_bytes_list.append(payload_block)
            
            # 3. Create Index Table Entry
            index_entries.append({
                'id': rec_id.bytes,
                'vector_offset': current_vector_offset,
                'payload_offset': current_payload_offset,
                'payload_len': payload_len,
                'status': status
            })
            
            current_vector_offset += single_vector_padded_size
            current_payload_offset += len(payload_block)
            
        # Compile index table binary
        index_table_bytes = b''
        for entry in index_entries:
            index_table_bytes += struct.pack(
                cls.INDEX_RECORD_FORMAT,
                entry['id'],
                entry['vector_offset'],
                entry['payload_offset'],
                entry['payload_len'],
                entry['status']
            )
            
        # Assemble index padding
        index_padding = b'\x00' * vector_padding_size
        
        # Assemble full body bytes
        vector_block_bytes = b''.join(vector_bytes_list)
        payload_block_bytes = b''.join(payload_bytes_list)
        
        body_bytes = index_table_bytes + index_padding + vector_block_bytes + payload_block_bytes
        
        # Calculate checksum over the entire body (excluding header)
        checksum = zlib.crc32(body_bytes) & 0xffffffff
        
        # Compile header binary
        header_bytes = struct.pack(
            cls.HEADER_FORMAT,
            b'VSEG',         # Magic Number
            1,              # Version
            segment_id.bytes,
            record_count,
            deleted_count,
            created_ts,
            checksum
        )
        
        return header_bytes + body_bytes

    @classmethod
    def deserialize(cls, data: bytes, dimension: int) -> Tuple[uuid.UUID, list[dict]]:
        """
        Deserialize segment binary bytes back into records.
        """
        if len(data) < 64:
            raise ValueError("Invalid segment data: Less than header size (64 bytes).")
            
        # Unpack header
        header_data = data[:64]
        magic, version, seg_id_bytes, record_count, deleted_count, created_ts, checksum = struct.unpack(
            cls.HEADER_FORMAT,
            header_data
        )
        
        if magic != b'VSEG':
            raise ValueError(f"Invalid magic number: {magic}. Not a valid segment file.")
            
        segment_id = uuid.UUID(bytes=seg_id_bytes)
        
        # Verify Checksum over body bytes
        body_bytes = data[64:]
        actual_checksum = zlib.crc32(body_bytes) & 0xffffffff
        if actual_checksum != checksum:
            raise ValueError(f"Segment checksum mismatch! Expected: {checksum}, Actual: {actual_checksum} (Data corrupted).")
            
        records = []
        single_vector_raw_size = dimension * 4
        single_vector_padded_size = (single_vector_raw_size + 63) // 64 * 64
        
        # Unpack index table
        index_table_offset = 64
        for i in range(record_count):
            entry_offset = index_table_offset + (i * 32)
            entry_data = data[entry_offset:entry_offset + 32]
            rec_id_bytes, v_offset, p_offset, p_len, status = struct.unpack(
                cls.INDEX_RECORD_FORMAT,
                entry_data
            )
            
            # Read vector
            v_bytes = data[v_offset:v_offset + single_vector_raw_size]
            vector = np.frombuffer(v_bytes, dtype=np.float32).tolist()
            
            # Read payload (offset points to length header)
            # Length header is 4 bytes
            payload_data_bytes = data[p_offset + 4:p_offset + 4 + p_len]
            payload_str = payload_data_bytes.decode('utf-8')
            payload = json.loads(payload_str)
            
            records.append({
                'id': uuid.UUID(bytes=rec_id_bytes),
                'vector': vector,
                'payload': payload,
                'status': status
            })
            
        return segment_id, records
