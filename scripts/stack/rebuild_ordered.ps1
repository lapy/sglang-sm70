# Rebuild the SM70 stack in 01..10 order from an already-resolved tree.
#
#   pwsh rebuild-ordered.ps1 -Repo <sglang-sm70> [-Source sm70/stack] [-Target sm70/ordered]
#
# Replaying each cluster's *file set* avoids re-resolving the environ.py merges
# that the incremental cherry-picks already settled.
param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [string]$Base = "sm70/base",
  [string]$Source = "sm70/stack",
  [string]$Target = "sm70/ordered",
# Paths that MUST exist in the rebuilt tree. Use this to pin the files a replay has to
# carry, so a stale -Source cannot silently produce a chain that is missing them.
[string[]]$ExpectPath = @()
)
Set-Location $Repo

$clusters = @(
  @{ Name = "01-kernels";      Re = '^python/sglang/kernels/' },
  @{ Name = "02-attention";    Re = '^python/sglang/srt/layers/attention/' },
  @{ Name = "03-moe-quant";    Re = '^python/sglang/srt/layers/(moe|quantization)/' },
  @{ Name = "04-models";       Re = '^python/sglang/srt/(models|configs)/|^python/sglang/srt/layers/(hyperconnection\.py|layer_boundary/|linear/|elementwise/|parameter\.py)' },
  @{ Name = "05-cache-runner"; Re = '^python/sglang/srt/(mem_cache|model_executor)/' },
  @{ Name = "06-speculative";  Re = '^python/sglang/srt/speculative/' },
  @{ Name = "07-env-mm-docs";  Re = '^python/sglang/srt/(environ\.py|arg_groups/|server_args\.py|multimodal/)|^python/sglang/multimodal_gen/|^docs/|^scripts/|^SM70\.md$' },
  @{ Name = "08-serving";      Re = '^python/sglang/srt/' },
  @{ Name = "09-build";        Re = '^(docker/|patches/|requirements\.txt|python/pyproject\.toml|NOTICE|README\.md|SECURITY\.md|\.gitignore|\.pre-commit-config\.yaml)' },
  @{ Name = "10-tests";        Re = '^test/' }
)

$status = @(git diff --name-status "$Base..$Source")
$assign = New-Object 'System.Collections.Generic.Dictionary[string,object]'
foreach ($line in $status) {
  $parts = $line -split "`t"
  $st = $parts[0]; $path = $parts[1]
  foreach ($c in $clusters) {
    if ($path -match $c.Re) {
      if (-not $assign.ContainsKey($c.Name)) { $assign[$c.Name] = @() }
      $assign[$c.Name] += ,@($st, $path)
      break
    }
  }
}

git checkout --quiet -B $Target $Base
if ($LASTEXITCODE -ne 0) { throw "cannot reset $Target" }

foreach ($c in $clusters) {
  $entries = $assign[$c.Name]
  if (-not $entries -or $entries.Count -eq 0) { Write-Host "skip $($c.Name)"; continue }
  $checkout = @(); $remove = @()
  foreach ($e in $entries) {
    if ($e[0] -eq 'D') { $remove += $e[1] } else { $checkout += $e[1] }
  }
  if ($remove.Count -gt 0) { git rm --quiet -- $remove 2>$null | Out-Null }
  if ($checkout.Count -gt 0) {
    $present = @()
    foreach ($one in $checkout) {
      git cat-file -e "${Source}:$one" 2>$null
      if ($LASTEXITCODE -eq 0) { $present += $one }
    }
    if ($present.Count -gt 0) {
      git checkout --quiet $Source -- $present
      if ($LASTEXITCODE -ne 0) { throw "checkout failed for $($c.Name)" }
    }
    if ($checkout.Count -ne $present.Count) {
      Write-Host "skip $($checkout.Count - $present.Count) path(s) deleted upstream in $($c.Name)"
    }
  }
  git add -A | Out-Null
  $msg = "$($c.Name): SM70 stack step (ordered rebuild from $Source)"
  git commit --quiet -m $msg
  if ($LASTEXITCODE -ne 0) { throw "commit failed for $($c.Name)" }
  Write-Host ("OK {0,-16} files={1}" -f $c.Name, $entries.Count)
}
Write-Host "tip: $(git log -1 --format='%h %s')"

# The replay must reproduce $Source exactly. A stale or wrong source (for example a branch
# that does not yet contain a commit made moments earlier) silently produces a chain that is
# "consistent" with the wrong tree, which is how 36 lines of documentation were lost once.
# Fail loudly instead of leaving that to a manual check.
git diff --quiet $Target $Source
if ($LASTEXITCODE -ne 0) {
  $stat = git diff --stat $Target $Source | Select-Object -Last 1
  $detail = "replay does not match $Source ($stat)"
  throw ($detail + ". Check that $Source contains every commit you expect, and that it is not stale relative to the branch you just committed to.")
}

# -File passes a comma-joined string rather than an array, so normalise both forms.
$ExpectPath = @($ExpectPath | ForEach-Object { $_ -split "," } |
                ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($ExpectPath.Count -gt 0) {
  $missing = @()
  foreach ($p in $ExpectPath) {
    git cat-file -e "${Target}:$p" 2>$null
    if ($LASTEXITCODE -ne 0) { $missing += $p }
  }
  if ($missing.Count -gt 0) {
    $detail = "rebuilt tree is missing expected path(s): " + ($missing -join ", ")
    throw ($detail + ". The source tree probably predates the change that added them.")
  }
  Write-Host ("content pinned: all {0} expected path(s) present" -f $ExpectPath.Count)
}
Write-Host "verified: $Target reproduces $Source exactly"
