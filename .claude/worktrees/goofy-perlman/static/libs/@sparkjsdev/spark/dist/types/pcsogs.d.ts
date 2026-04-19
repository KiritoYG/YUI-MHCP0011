import { SplatEncoding } from './PackedSplats';
import { PcSogsJson, PcSogsV2Json } from './SplatLoader';
export declare function unpackPcSogs(json: PcSogsJson | PcSogsV2Json, extraFiles: Record<string, ArrayBuffer>, splatEncoding: SplatEncoding): Promise<{
    packedArray: Uint32Array;
    numSplats: number;
    extra: Record<string, unknown>;
}>;
export declare function unpackPcSogsZip(fileBytes: Uint8Array, splatEncoding: SplatEncoding): Promise<{
    packedArray: Uint32Array;
    numSplats: number;
    extra: Record<string, unknown>;
}>;
