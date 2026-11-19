# 강수 비교 실험용 입력 필터 (제출 코드는 수정하지 않음; model_base.Model을 그대로 감쌈)
from model_base import Model as _Base
PANEL_WX = ("precip_mm", "temp_avg")
PANEL_PRECIP = ("precip_mm",)
def make(drop_panel, drop_asos, keep_temp=False):
    class Model(_Base):
        def _rows(self, rows):
            cols = PANEL_PRECIP if keep_temp else PANEL_WX
            return [dict(r, **{k: "" for k in cols}) for r in rows] if drop_panel else rows
        def _aux(self, aux):
            # ASOS를 빼면 aux 전체를 넘기지 않는다(남는 품질주의표는 제출 코드가 쓰지 않음).
            # 빈 ASOS를 넘기면 강수 특징 열이 전부 결측이 되어 HGB가 오류를 내므로, 열 자체를 빼는 방식으로 비교한다.
            return None if drop_asos else aux
        def fit(self, rows, national, aux=None):
            return super().fit(self._rows(rows), national, self._aux(aux))
        def predict(self, rows, origin_week, national, aux=None):
            return super().predict(self._rows(rows), origin_week, national, self._aux(aux))
    return Model
