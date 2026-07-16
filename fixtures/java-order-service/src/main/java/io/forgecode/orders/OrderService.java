package io.forgecode.orders;

import java.math.BigDecimal;
import java.util.List;

public final class OrderService {
    public BigDecimal calculateTotal(List<OrderItem> items) {
        BigDecimal total = BigDecimal.ZERO;
        for (OrderItem item : items) {
            total = total.add(item.getUnitPrice());
        }
        return total;
    }
}
