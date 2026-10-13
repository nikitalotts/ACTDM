#!/bin/bash
# Статистика по проекту Slurm: выделено / потрачено / осталось
# Использование: bash projstat.sh proj_1891 [дата_начала, по умолчанию 2019-01-01]
# Лимиты берутся из sacctmgr, расход из sreport.

PROJ=${1:-proj_1891}
START=${2:-2019-01-01}

# Название проекта (поле Description аккаунта)
DESC=$(sacctmgr -n -P show account "$PROJ" format=Description 2>/dev/null | head -n1)
[ -z "$DESC" ] || [ "$DESC" = "$PROJ" ] && DESC="(название не указано)"

# Лимит проекта (строка без пользователя)
LIM=$(sacctmgr -n -P show assoc account="$PROJ" format=User,GrpTRESMins 2>/dev/null \
      | awk -F'|' '$1=="" && $2!="" {print $2; exit}')

# Расход в часах по ресурсам и пользователям
USAGE=$(sreport -n -P -t Hours cluster AccountUtilizationByUser \
        Accounts="$PROJ" Start="$START" End=now --tres=cpu,gres/gpu 2>/dev/null)

if [ -z "$LIM" ] && [ -z "$USAGE" ]; then
  echo "Нет данных по проекту $PROJ (проверьте ID командой mp)"
  exit 1
fi

printf '%s\n' "LIM|$LIM" "$USAGE" | awk -F'|' -v proj="$PROJ" -v start="$START" -v desc="$DESC" '
function name(t) { return t == "cpu" ? "CPU" : (t == "gres/gpu" ? "GPU" : t) }
function bar(p,   n, s, i) {
  n = int(p / 5); if (n > 20) n = 20; if (n < 0) n = 0
  s = "["; for (i = 0; i < 20; i++) s = s (i < n ? "#" : "."); return s "]"
}
$1 == "LIM" {
  n = split($2, parts, ",")
  for (i = 1; i <= n; i++) { split(parts[i], kv, "="); if (kv[1] != "") L[kv[1]] = kv[2] / 60 }
  next
}
NF >= 6 {
  login = $3; tres = $5; used = $6 + 0
  if (login == "") T[tres] = used
  else { if (!(login in seen)) { seen[login] = 1; users[++nu] = login }; UU[login, tres] = used }
}
END {
  print ""
  print "Проект:   " proj
  print "Название: " desc
  print "Период:   расход с " start ", все значения в часах"
  print "--------------------------------------------------------------------------"
  print "Ресурс     Выделено    Потрачено     Осталось       %"
  print "--------------------------------------------------------------------------"
  for (t in L) {
    a = L[t]; u = T[t] + 0; r = a - u; p = (a > 0) ? u / a * 100 : 0
    printf "%-6s %12.1f %12.1f %12.1f %6.1f%%  %s\n", name(t), a, u, r, p, bar(p)
    done[t] = 1
  }
  for (t in T) if (!(t in done))
    printf "%-6s %12s %12.1f %12s\n", name(t), "без лимита", T[t], "-"

  print ""
  print "Расход по участникам (часы):"
  print "--------------------------------------"
  print "Пользователь            CPU        GPU"
  print "--------------------------------------"
  for (i = 1; i <= nu; i++)
    printf "%-16s %10.1f %10.1f\n", users[i], UU[users[i], "cpu"], UU[users[i], "gres/gpu"]
  print ""
}'
