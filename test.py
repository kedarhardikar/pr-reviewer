import git

repo = git.Repo(r"D:\tmp\pr-reviewer\ibm_hackathon")
base = "3e71c597397af68c01f1b3ef01198ebce770b879"
head = "1e8e6049601fbd44b470347b602ae1e858d0023c"

diff_index = repo.commit(base).diff(head, create_patch=True)
for d in diff_index:
    print(f"path={d.b_path or d.a_path!r}")
    print(f"  d.diff type={type(d.diff)}  len={len(d.diff) if d.diff else 'None'}")
    print(f"  new={d.new_file} deleted={d.deleted_file} renamed={d.renamed_file}")
    print()