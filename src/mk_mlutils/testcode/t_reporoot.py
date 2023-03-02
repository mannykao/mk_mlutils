from pathlib import Path, PurePosixPath
from mk_mlutils import projconfig

kRepoRoot="mk_mlutils"
kToSrcRoot="src/mk_mlutils"


if __name__ == '__main__':
	#projconfig.setRepoRoot("mk_mlutils/src/mk_mlutils", __file__)

	print(f"{__file__=}")

	print(f"{projconfig.getRefFile()=}")
	print(f"{projconfig.getRepoRoot()=}")
	print(f"{projconfig.getDataFolder()=}")

