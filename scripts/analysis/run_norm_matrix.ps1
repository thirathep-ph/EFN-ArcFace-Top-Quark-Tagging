$root = "D:\research\jet_tagging"
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"
$log = "$root\experiments\physics_sr\norm_matrix_run.log"
Start-Transcript -Path $log -Append
try {
  $norms = @(@("z", "z"), @("sigma", "sigma"), @("robust", "robust"),
              @("raw", "raw"), @("minmax", "mm"))
  foreach ($pair in $norms) {
    $norm, $short = $pair[0], $pair[1]
    foreach ($seed in @(42, 123, 7)) {
      $tag = "norm_${short}_s$seed"
      Write-Output "[matrix] fit start norm=$norm seed=$seed $(Get-Date -Format o)"
      & $py -m scripts.analysis.fit_physics_sr --mode quick --seed $seed `
          --norm $norm --no-nest-tanh --out-tag $tag
      if ($?) {
        Write-Output "[matrix] eval-self start $tag $(Get-Date -Format o)"
        & $py -m scripts.analysis.compute_symbolic_eval `
            --fit "fit_quick_${tag}.json" --out "eval_${tag}_self.json"
        Write-Output "[matrix] eval-full start $tag $(Get-Date -Format o)"
        & $py -m scripts.analysis.compute_symbolic_eval `
            --fit "fit_quick_${tag}.json" --out "eval_${tag}_full.json" --eval full
      }
      Write-Output "[matrix] done norm=$norm seed=$seed ok=$? $(Get-Date -Format o)"
    }
  }
} finally {
  Stop-Transcript
}
