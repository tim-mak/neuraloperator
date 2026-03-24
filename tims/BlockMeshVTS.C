/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    BlockMeshStructured

Description
    Reads a blockMeshDict and writes each block as a VTK XML Structured Grid
    (.vts) file, preserving the per-block i/j/k topology.

    Also writes a VTK MultiBlock Dataset (.vtm) linking all block files so
    ParaView / VisIt open them as a single dataset.

    All blockMeshDict transforms (prescale, scale, coordinate-system rotation
    and translation) are applied before writing.

Usage
    BlockMeshStructured [OPTIONS]

    Options:
      -region <name>   Mesh region (default: region0 / defaultRegion)
      -output  <dir>   Output directory (default: constant/blockMeshVTK)

Notes
    Point ordering inside block::points() is
        index = i + (ni+1)*(j + (nj+1)*k)
    i.e. i varies fastest, which is identical to the VTK StructuredGrid
    convention, so points can be written sequentially with no reordering.

\*---------------------------------------------------------------------------*/

#include "Time.H"
#include "IOdictionary.H"
#include "blockMesh.H"
#include "argList.H"
#include "OFstream.H"
#include "OSspecific.H"
#include "treeBoundBox.H"
#include "treeDataPoint.H"
#include "indexedOctree.H"

using namespace Foam;

// * * * * * * * * * * * * * Local helper functions  * * * * * * * * * * * * //

//- Write one block as VTK XML StructuredGrid (.vts)
//
//  pts   : per-block point field, already transformed to global coordinates
//          (obtained via blockMesh::globalPosition(blk.points()))
void writeVTS
(
    const fileName&   outDir,
    const label       blockIdx,
    const block&      blk,
    const pointField& pts
)
{
    const label ni = blk.density().x();   // cell count in i
    const label nj = blk.density().y();   // cell count in j
    const label nk = blk.density().z();   // cell count in k

    const fileName vtsFile
    (
        outDir / ("block_" + Foam::name(blockIdx) + ".vts")
    );
    OFstream os(vtsFile);

    Info<< "  Writing block " << blockIdx
        << "  density (" << ni << " x " << nj << " x " << nk << ")"
        << "  -> " << vtsFile.name() << nl;

    // ------------------------------------------------------------------ //
    // VTK XML StructuredGrid format
    // Extent:  0..ni, 0..nj, 0..nk  (corner indices, not cell counts)
    // Points:  (ni+1)*(nj+1)*(nk+1) Float64 triplets, i-fastest order
    // ------------------------------------------------------------------ //

    os  << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"StructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\">\n"
        << "  <StructuredGrid WholeExtent=\""
           "0 " << ni << " 0 " << nj << " 0 " << nk << "\">\n"
        << "    <Piece Extent=\""
           "0 " << ni << " 0 " << nj << " 0 " << nk << "\">\n"
        << "      <Points>\n"
        << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"ascii\">\n";

    // block::points() stores points in exactly the VTK structured order:
    //   linear index = i + (ni+1)*(j + (nj+1)*k)   { i fastest }
    // We can therefore iterate sequentially through the array.
    const label nPts = (ni + 1) * (nj + 1) * (nk + 1);

    // FIX: Force double-precision output for ASCII text
    os.precision(16);

    for (label idx = 0; idx < nPts; ++idx)
    {
        const point& p = pts[idx];
        os  << "          "
            << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    // Optional: Reset precision back to default if writing other things
    os.precision(6);

    os  << "        </DataArray>\n"
        << "      </Points>\n";

    // ---- optional: write block index as cell data ---- //
    const label nCells = ni * nj * nk;

    os  << "      <CellData Scalars=\"blockIndex\">\n"
        << "        <DataArray type=\"Int32\" Name=\"blockIndex\""
           " format=\"ascii\">\n"
        << "          ";
    for (label c = 0; c < nCells; ++c)
    {
        os << blockIdx;
        os << (((c + 1) % 20 == 0) ? "\n          " : " ");
    }
    os  << "\n"
        << "        </DataArray>\n"
        << "      </CellData>\n"
        << "    </Piece>\n"
        << "  </StructuredGrid>\n"
        << "</VTKFile>\n";
}


//- Write a VTK MultiBlock Dataset (.vtm) referencing all block .vts files
void writeVTM
(
    const fileName& outDir,
    const blockMesh& blocks
)
{
    const fileName vtmFile(outDir / "blockMesh.vtm");
    OFstream os(vtmFile);

    Info<< nl << "Writing multiblock index -> " << vtmFile << nl;

    os  << "<?xml version=\"1.0\"?>\n"
        << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\""
           " byte_order=\"LittleEndian\">\n"
        << "  <vtkMultiBlockDataSet>\n";

    forAll(blocks, bi)
    {
        const word blkName =
            blocks[bi].zoneName().empty()
          ? word("block_" + Foam::name(bi))
          : blocks[bi].zoneName();

        os  << "    <DataSet index=\"" << bi << "\""
            << " name=\"" << blkName << "\""
            << " file=\"block_" << bi << ".vts\""
            << "/>\n";
    }

    os  << "  </vtkMultiBlockDataSet>\n"
        << "</VTKFile>\n";
}


// * * * * * * * * * * * * * * * * Main  * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Read blockMeshDict and write each block as a VTK XML "
        "StructuredGrid (.vts) file, preserving i/j/k structured topology.\n"
        "A VTK MultiBlock Dataset file (blockMesh.vtm) is also written."
    );

    argList::addOption
    (
        "region",
        "name",
        "Specify alternative mesh region name (default: region0)"
    );

    argList::addOption
    (
        "output",
        "dir",
        "Output directory for .vts/.vtm files (default: constant/blockMeshVTK)"
    );

    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"

    // ------------------------------------------------------------------ //
    // Region / dictionary
    // ------------------------------------------------------------------ //

    const word regionName =
        args.getOrDefault<word>("region", polyMesh::defaultRegion);

    // Locate blockMeshDict in system/ (or system/<region>/)
    const fileName dictPath
    (
        runTime.system()
      / (regionName == polyMesh::defaultRegion ? "" : regionName)
      / "blockMeshDict"
    );

    IOdictionary meshDict
    (
        IOobject
        (
            "blockMeshDict",
            runTime.system(),
            regionName == polyMesh::defaultRegion ? "" : regionName,
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        )
    );

    Info<< "Reading blockMeshDict: "
        << meshDict.objectRelPath() << nl << endl;

    // ------------------------------------------------------------------ //
    // Build the blockMesh topology (reads vertices, edges, blocks, patches)
    // ------------------------------------------------------------------ //

    blockMesh blocks(meshDict, regionName);

    if (!blocks.good())
    {
        FatalErrorInFunction
            << "blockMesh failed: topology not constructed"
            << exit(FatalError);
    }

    const label nBlocks = blocks.size();

    Info<< "Blocks      : " << nBlocks << nl
        << "Transforms  : "
        << (blocks.hasPointTransforms() ? "yes" : "none") << nl
        << endl;

    // ------------------------------------------------------------------ //
    // Output directory
    // ------------------------------------------------------------------ //

    fileName outDir =
        args.getOrDefault<fileName>
        (
            "output",
            runTime.constant() / "blockMeshVTK"
        );

    mkDir(outDir);

    Info<< "Output dir  : " << outDir << nl << endl;

    // ------------------------------------------------------------------ //
    // Write one .vts per block
    // ------------------------------------------------------------------ //
    // 1. Get the final, transformed global points
    const pointField& globalPts = blocks.points();

    Info<< "Building Octree for " << globalPts.size() << " global points..." << endl;

    // 2. Build the Bounding Box and Tree Data
    treeBoundBox overallBb(globalPts);
    treeDataPoint treeData(globalPts);

    // 3. Construct the Indexed Octree for O(log N) searches
    indexedOctree<treeDataPoint> pointTree
    (
        treeData,
        overallBb,
        10,     // maxLevel (max depth of the tree)
        10,     // leafsize (max points per leaf)
        3.0     // duplicity
    );

    Info<< "Octree built. Snapping VTS coordinates..." << nl << endl;

    for (label bi = 0; bi < nBlocks; ++bi)
    {
        const block& blk = blocks[bi];

        const label ni = blk.density().x();
        const label nj = blk.density().y();
        const label nk = blk.density().z();
        const label nPts = (ni + 1) * (nj + 1) * (nk + 1);

        pointField localPts(nPts);

        // Get the mathematically rebuilt analytical points
        tmp<pointField> analyticalPts = blocks.globalPosition(blk.points());

        // We know the geometric drift is microscopic (e.g. 1e-6).
        // Setting a search radius of 1e-3 meters is extremely safe and keeps the search lightning fast.
        const scalar searchRadiusSq = Foam::sqr(1e-3);

        for (label i = 0; i < nPts; ++i)
        {
            const point& pAnalytic = analyticalPts()[i];

            // 4. Query the Octree
            pointIndexHit info = pointTree.findNearest
            (
                pAnalytic,
                searchRadiusSq
            );

            if (info.hit())
            {
                // Snap exactly to the true OpenFOAM 64-bit coordinate
                localPts[i] = globalPts[info.index()];
            }
            else
            {
                // Fallback (Should never happen unless blockMeshDict is fundamentally broken)
                localPts[i] = pAnalytic;
            }
        }
        writeVTS(outDir, bi, blk, localPts);
    }

    // ------------------------------------------------------------------ //
    // Write the multiblock index (.vtm)
    // ------------------------------------------------------------------ //

    writeVTM(outDir, blocks);

    Info<< nl << "End" << nl << endl;
    return 0;
}


// ************************************************************************* //
