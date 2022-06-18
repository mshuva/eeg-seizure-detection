function metrics = perfMetric(C)
    TP = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TN = C(2,2);
    
    sensitivity = TP/(TP + FN);
    specificity = TN/(FP + TN);
    precision = TP/(TP + FP);
    fallout = 1 - specificity;
    f1 = 2*TP/(2*TP + FP + FN);
    
    metric = [sensitivity; specificity; precision; fallout; f1]
end