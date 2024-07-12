import numpy as np
import scipy.stats as st


# data of goals scored by 20 footballers in a calendar year



def Confidence_Interval(fb_data, distribution, confidence_level):
    """
    distribution: "T" or "N"
    """
    def format_num(num):
        return round(num * 100, 2)
    input_dict = {
        "confidence": confidence_level,
        "loc": np.mean(fb_data),
        "scale": st.sem(fb_data),
    }
    if distribution == "T":
        dist_obj = st.t
        input_dict.update({"df": len(fb_data) - 1})
    else:
        dist_obj = st.norm
    # 置信水平create 95% confidence interval
    # confidence_level = 0.95
    # t.interval() 计算置信区间 （n<30）
    lower, upper = dist_obj.interval(**input_dict)
    lower = format_num(lower)
    upper = format_num(upper)
    print(f"{distribution}:({lower},{upper})")
    return lower, upper


def main():
    fb_data = [
        0.8620689655172413,
        0.853448275862069,
        0.8793103448275862,
        0.8706896551724138,
        0.8706896551724138,
        0.8706896551724138,
        0.8706896551724138,
        0.8793103448275862,
        0.853448275862069,
        0.8620689655172413,
    ]
    Confidence_Interval(fb_data, distribution="T", confidence_level=0.95)
    Confidence_Interval(fb_data, distribution="N", confidence_level=0.95)


if __name__ == "__main__":
    main()
